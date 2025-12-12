# coding: utf-8
# tool/generate_user_matrix.py
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

# Hypergraph modules

class HypergraphBuilder:
    def __init__(self, n_users, n_items, edge_index, user_graph_dict, mm_adj, knn_k=20, max_pool_size=200000):
        self.n_users = n_users
        self.n_items = n_items
        if isinstance(edge_index, torch.Tensor):
            self.edge_index = edge_index.detach().cpu()
        else:
            self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.user_graph_dict = user_graph_dict
        self.mm_adj = mm_adj.coalesce()
        self.knn_k = knn_k
        self.max_pool_size = max_pool_size
        self.temperature = temperature

        self._build_item_knn()
        self.E_uis, self.E_uui = self._build_hyperedges()

    def _build_item_knn(self):
        indices = self.mm_adj.indices()  # [2, nnz]
        rows, cols = indices[0], indices[1]
        item_knn = [[] for _ in range(self.n_items)]
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r != c:
                item_knn[r].append(c)
        for i in range(self.n_items):
            if len(item_knn[i]) > self.knn_k:
                item_knn[i] = item_knn[i][:self.knn_k]
        self.item_knn = item_knn

    def _build_hyperedges(self):
        edges = self.edge_index.t().tolist()
        uv_pairs = []
        for u, vi in edges:
            if u < self.n_users and vi >= self.n_users:
                uv_pairs.append((u, vi - self.n_users))

        # E_uis: (u, vj, vk)
        E_uis = []
        for u, vj in uv_pairs:
            knns = self.item_knn[vj]
            for vk in knns:
                if vk != vj:
                    E_uis.append((u, vj, vk))

        # E_uui: (ui, uj, v)
        E_uui = []
        for u, v in uv_pairs:
            if u in self.user_graph_dict:
                nbrs = self.user_graph_dict[u][0]
                for uj in nbrs:
                    if uj != u:
                        E_uui.append((u, uj, v))

        if len(E_uis) > self.max_pool_size:
            E_uis = E_uis[:self.max_pool_size]
        if len(E_uui) > self.max_pool_size:
            E_uui = E_uui[:self.max_pool_size]
        return E_uis, E_uui


class HypergraphGlobalSampler:
    def __init__(self, E_uis, E_uui, p_uis=0.5):
        self.E_uis = E_uis
        self.E_uui = E_uui
        self.p_uis = p_uis

    def sample_batch(self, B):
        num_uis = int(B * self.p_uis)
        num_uui = B - num_uis
        batch_uis = random.sample(self.E_uis, num_uis) if len(self.E_uis) >= num_uis else self.E_uis
        batch_uui = random.sample(self.E_uui, num_uui) if len(self.E_uui) >= num_uui else self.E_uui
        E_batch = [('uis', e) for e in batch_uis] + [('uui', e) for e in batch_uui]

        users = set()
        items = set()
        for t, e in E_batch:
            if t == 'uis':
                u, vj, vk = e
                users.add(u); items.add(vj); items.add(vk)
            else:
                ui, uj, v = e
                users.add(ui); users.add(uj); items.add(v)
        return E_batch, users, items


class TypeAwareHGATLayer(nn.Module):
    def __init__(self, dim, num_types=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.WQ = nn.Linear(dim, dim, bias=False)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.type_embed = nn.Embedding(num_types, dim)  # τ ∈ {uis:0, uui:1}
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_user, z_item, users_set, items_set, E_batch):

        device = z_user.device
        d = z_user.size(1)

        users = sorted(list(users_set))
        items = sorted(list(items_set))
        u_map = {u:i for i,u in enumerate(users)}
        i_map = {v:j for j,v in enumerate(items)}

        Zu = z_user[users]  # [Nu, d]
        Zi = z_item[items]  # [Ni, d]

        Eh_type = []
        H_members = []
        for (t, e) in E_batch:
            if t == 'uis':
                u, vj, vk = e
                if (u in u_map) and (vj in i_map) and (vk in i_map):
                    Eh_type.append(0)  # uis
                    H_members.append(('uis', (u_map[u], i_map[vj], i_map[vk])))
            else:
                ui, uj, v = e
                if (ui in u_map) and (uj in u_map) and (v in i_map):
                    Eh_type.append(1)  # uui
                    H_members.append(('uui', (u_map[ui], u_map[uj], i_map[v])))

        if len(H_members) == 0:
            du = torch.zeros_like(Zu)
            di = torch.zeros_like(Zi)
            return users, items, du, di

        M = len(H_members)
        Wtau = self.type_embed(torch.tensor(Eh_type, device=device))  # [M, d]

        H_init = torch.zeros(M, d, device=device)
        for idx, (t, mem) in enumerate(H_members):
            if t == 'uis':
                uu, vj, vk = mem
                H_init[idx] = (Zu[uu] + Zi[vj] + Zi[vk]) / 3.0
            else:
                ui_, uj_, v_ = mem
                H_init[idx] = (Zu[ui_] + Zu[uj_] + Zi[v_]) / 3.0

        Q_u = self.WQ(Zu)
        Q_i = self.WQ(Zi)
        K_h = self.WK(H_init) + Wtau
        V_h = self.WV(H_init)

        u2h = [[] for _ in range(len(users))]
        i2h = [[] for _ in range(len(items))]
        for h_idx, (t, mem) in enumerate(H_members):
            if t == 'uis':
                uu, vj, vk = mem
                u2h[uu].append(h_idx)
                i2h[vj].append(h_idx); i2h[vk].append(h_idx)
            else:
                ui_, uj_, v_ = mem
                u2h[ui_].append(h_idx); u2h[uj_].append(h_idx)
                i2h[v_].append(h_idx)

        du = torch.zeros_like(Zu)
        di = torch.zeros_like(Zi)
        scale = (d ** 0.5)

        for uu in range(len(users)):
            hs = u2h[uu]
            if not hs: continue
            q = Q_u[uu:uu+1]      # [1, d]
            k = K_h[hs]           # [H, d]
            v = V_h[hs]           # [H, d]
            att = torch.softmax((q @ k.t())/scale, dim=-1)
            msg = att @ v
            du[uu] = self.dropout(msg.squeeze(0))

        for vv in range(len(items)):
            hs = i2h[vv]
            if not hs: continue
            q = Q_i[vv:vv+1]
            k = K_h[hs]
            v = V_h[hs]
            att = torch.softmax((q @ k.t())/scale, dim=-1)
            msg = att @ v
            di[vv] = self.dropout(msg.squeeze(0))

        return users, items, du, di


class CGTC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CGTC, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 20
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.dataset = dataset
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 64
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.mm_adj = None

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)).to(self.device)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.v_feat)  # 256)
        if self.t_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)).to(self.device)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.t_feat)

        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.hg_dim = config['hg_dim'] if 'hg_dim' in config else self.dim_latent
        self.hg_batch_edges = config['hg_batch_edges'] if 'hg_batch_edges' in config else 512
        self.hg_p_uis = config['hg_p_uis'] if 'hg_p_uis' in config else 0.5
        self.hg_alpha = config['hg_alpha'] if 'hg_alpha' in config else 0.2
        self.hg_layers = config['hg_layers'] if 'hg_layers' in config else 1

        mm_adj_cpu = self.mm_adj.detach().cpu() if isinstance(self.mm_adj, torch.Tensor) else self.mm_adj
        edge_index_cpu = self.edge_index.detach().cpu()
        self.hg_builder = HypergraphBuilder(
            n_users=self.n_users,
            n_items=self.n_items,
            edge_index=edge_index_cpu,
            user_graph_dict=self.user_graph_dict,
            mm_adj=mm_adj_cpu,
            knn_k=self.knn_k
        )
        self.hg_sampler = HypergraphGlobalSampler(
            self.hg_builder.E_uis, self.hg_builder.E_uui, p_uis=self.hg_p_uis
        )
        self.hg_layers_mod = nn.ModuleList([TypeAwareHGATLayer(self.hg_dim, num_types=2, dropout=0.1)
                                            for _ in range(self.hg_layers)])


        in_dim = self.dim_latent
        if self.construction == "cat" and self.v_feat is not None and self.t_feat is not None:
            in_dim = self.dim_latent * 2

        self.user_proj_to_hg = nn.Linear(in_dim, self.hg_dim)
        self.item_proj_to_hg = nn.Linear(in_dim, self.hg_dim)
        self.user_proj_from_hg = nn.Linear(self.hg_dim, in_dim)
        self.item_proj_from_hg = nn.Linear(self.hg_dim, in_dim)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        representation = None

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            representation = self.v_rep
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            if representation is None:
                representation = self.t_rep
            else:
                if self.construction == 'cat':
                    representation = torch.cat((self.v_rep, self.t_rep), dim=1)
                else:
                    representation += self.t_rep

        if self.construction == 'weighted_sum':
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)

        if self.construction == 'weighted_max':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.max(user_rep, dim=2).values

        if self.construction == 'cat':
            if self.v_rep is not None:
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep
                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        item_rep = representation[self.num_user:]

        h = item_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj.to(self.device), h)
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep = user_rep + h_u1
        item_rep = item_rep + h

        E_batch, users_set, items_set = self.hg_sampler.sample_batch(self.hg_batch_edges)

        z_user_hg = self.user_proj_to_hg(user_rep)   # [U, hg_dim]
        z_item_hg = self.item_proj_to_hg(item_rep)   # [I, hg_dim]

        du_acc = torch.zeros_like(z_user_hg)
        di_acc = torch.zeros_like(z_item_hg)
        for layer in self.hg_layers_mod:
            users, items, du, di = layer(z_user_hg, z_item_hg, users_set, items_set, E_batch)
            du_acc[users] += du
            di_acc[items] += di
            z_user_hg[users] = z_user_hg[users] + du
            z_item_hg[items] = z_item_hg[items] + di

        du_back = self.user_proj_from_hg(du_acc)
        di_back = self.item_proj_from_hg(di_acc)

        user_rep = user_rep.clone()
        item_rep = item_rep.clone()
        if len(users) > 0:
            user_rep[users] = user_rep[users] + self.hg_alpha * du_back[users]
        if len(items) > 0:
            item_rep[items] = item_rep[items] + self.hg_alpha * di_back[items]

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)

        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()

        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            if self.t_feat is not None:
                t_feat_target = F.dropout(t_feat_online.clone(), self.dropout)
            if self.v_feat is not None:
                v_feat_target = F.dropout(v_feat_online.clone(), self.dropout)

        loss_tv, loss_vt = 0.0, 0.0
        pos_item_nodes = torch.clamp(interaction[1], 0, self.n_items - 1)

        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[pos_item_nodes]
            t_feat_target = t_feat_target[pos_item_nodes]

            sim_t = cosine_similarity(
                t_feat_online,
                t_feat_target.detach(),
                dim=-1
            )
            loss_tv = 1 - (sim_t / self.temperature).mean()

        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[pos_item_nodes]
            v_feat_target = v_feat_target[pos_item_nodes]

            sim_v = cosine_similarity(
                v_feat_online,
                v_feat_target.detach(),
                dim=-1
            )
            loss_vt = 1 - (sim_v / self.temperature).mean()

        total_loss = (
                loss_value * self.loss_value_weight
                + (loss_tv + loss_vt) * self.contrastive_loss_weight
                + reg_loss
        )

        return total_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = h + x + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

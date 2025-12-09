import os
import argparse
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CGTC')
    parser.add_argument('--dataset', '-d', type=str, default='clothing')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/CGTC.yaml'
    )

    args = parser.parse_args()

    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config],  
        save_model=True
    )

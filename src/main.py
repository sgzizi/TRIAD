
import os
import argparse
from utils_package.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '24'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='TRIAD', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', type=int, default=0)

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
    }
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)



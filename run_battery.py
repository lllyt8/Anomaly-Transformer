import os
import argparse
from solver import Solver

def get_args():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='BATTERY')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    
    # Model parameters
    parser.add_argument('--win_size', type=int, default=15)  # 改为15
    parser.add_argument('--input_c', type=int, default=12)
    parser.add_argument('--output_c', type=int, default=12)
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    
    return parser.parse_args()

def get_config(args):
    config = {
        'mode': args.mode,
        'dataset': args.dataset,
        'data_path': args.data_path,
        'model_save_path': args.model_save_path,
        'win_size': args.win_size,
        'input_c': args.input_c,
        'output_c': args.output_c,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'anormly_ratio': args.anormly_ratio,
    }
    return config

def main():
    args = get_args()
    config = get_config(args)
    
    print('Start training...')
    solver = Solver(config)
    
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        print(f"Error: mode {args.mode} not recognized. Use 'train' or 'test'")

if __name__ == '__main__':
    main()

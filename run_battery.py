import os
import argparse
from solver import Solver
from utils.utils import *

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-4)  # Learning Rate
    parser.add_argument('--num_epochs', type=int, default=10)  # Epochs
    parser.add_argument('--d_model', type=int, default=512)  # Dimension of the model
    parser.add_argument('--k', type=int, default=3)  # Number of attention heads
    parser.add_argument('--win_size', type=int, default=100)  # Window size
    parser.add_argument('--input_c', type=int, default=12)  # input dimension
    parser.add_argument('--output_c', type=int, default=12)  # output dimension, same as input dimension
    parser.add_argument('--batch_size', type=int, default=64)  # batch size for cpu training
    # Save model Path
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='BATTERY')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='/Users/orient/Documents/Projects/GitHub/Anomaly-Transformer/data_factory/raw/DanDLN_Data.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    
    config = parser.parse_args()
    
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
        
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    solver = Solver(vars(config))
    
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    main()

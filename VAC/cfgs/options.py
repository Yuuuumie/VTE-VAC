import argparse
import shutil
import os

def parse_args():
    descript = 'Pytorch Implementation of vEGDnet'
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--num_classes', type=int, default=54)

    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='./models/EGD')
    parser.add_argument('--output_path', type=str, default='./outputs/EGD')
    parser.add_argument('--log_path', type=str, default='./logs/EGD')
    parser.add_argument('--modal', type=str, default='rgb', choices=['fusion', 'I3D','multitask', 'roi', 'rgb', 'Cholec80', 'THUMOS14'])
    parser.add_argument('--lambdas', type=str, default='[1,1]')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_lr', type=float, default=1e-5)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--model', type=str, default=None, help='the model choose to use')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--long_memory_length', type=int, default=1024)
    parser.add_argument('--long_memory_sampling_rate', type=int, default=4)
    parser.add_argument('--long_mask_ratio', type=float, default=0.75)
    parser.add_argument('--work_memory_length', type=int, default=256)
    parser.add_argument('--work_memory_sampling_rate', type=int, default=1)
    parser.add_argument('--work_mask_ratio', type=float, default=0.75)


    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
        
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args

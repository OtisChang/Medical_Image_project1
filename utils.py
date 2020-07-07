import argparse


def parse_args():
    desc = 'PyTorch example code for Kaggle competition -- Plant Seedlings Classification.\n' \
           'See https://www.kaggle.com/c/plant-seedlings-classification'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-r', '--root', default='./train', help='path to dataset')
    parser.add_argument('-w', '--weight', default='./resNet101_Pretrain_100e_5for1_lr0.001', help='path to model weights')
    parser.add_argument('-c', '--cuda_devices',type=int, default=1, help='path to model weights')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='input your batch size')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='input your number workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='input your learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='input your epochs')
    return parser.parse_args()

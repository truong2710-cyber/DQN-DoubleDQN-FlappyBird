import os
import argparse
import time
import torch
from hparams import HParams
from model import NeuralNetwork, init_weights
from dqn import train, test

def parse_args():
    """Parse hyper-parameters."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--logs_path', dest='logs_path', type=str,
                        help='path of the checkpoint folder', default='./logs')
    #parser.add_argument('-v', '--video_path', dest='video_path', type=str,
    #                    help='path of the video folder', default='./video')
    parser.add_argument('-r', '--restore', dest='restore', type=str,
                        help='restore checkpoint', default=None)
    parser.add_argument('-t', '--train', dest='train', type=bool,
                        help='train policy or not', default=False)
    parser.add_argument('-tp', '--type', dest='type', type=str,
                        help='dqn or double dqn', default='doubledqn')

    # directory
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str,
                        default='./checkpoint', help='saving directory')
    parser.add_argument('--hparam_path', type=str,
                        default='hparams.json',
                        help='hparam file')

    # parse the arguments
    args = parser.parse_args()

    return args


def parse_hparam(hparam_path):
    """Parse hyper-parameters."""
    params = HParams(hparam_path)
    return params


def main():
    """Main pipeline for Deep Reinforcement Learning with DQN and Double DQN."""
    args = parse_args()
    hparams = parse_hparam(args.hparam_path)
    print("Hyper-parameters parsed successfully!")

    # start training
    if args.train:
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)

        q_model = NeuralNetwork()
        target_model = NeuralNetwork()
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            q_model = q_model.cuda()
            target_model = target_model.cuda()

        q_model.apply(init_weights)
        start = time.time()

        if args.type == 'dqn':
            train(q_model, start, args, hparams, mode='dqn')
        elif args.type == 'doubledqn':
            train(q_model, start, args, hparams, target_model, mode='doubledqn')
    else:
        q_model = torch.load(args.restore,
                             map_location='cpu' if not torch.cuda.is_available() else None).eval()

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            q_model = q_model.cuda()

        test(q_model)


if __name__ == '__main__':
    main()
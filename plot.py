import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_path', dest='logs_path', type=str,
                        help='path of the log folder', default='./logs')
    args = parser.parse_args()

    return args

args = parse_args()

def main():
    """Plot."""
    args = parse_args()
    logs_path = args.logs_path

    # read in reward
    episode1, reward1 = zip(*np.load(os.path.join(logs_path, 'reward.npy')))
    episode2, reward2 = zip(*np.load(os.path.join(logs_path, 'reward_double.npy')))

    avg_reward1 = np.cumsum(reward1) / np.arange(1, len(reward1) + 1)
    avg_reward2 = np.cumsum(reward2) / np.arange(1, len(reward2) + 1)

    # subplot
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.set_xlabel('Episodes', fontsize=20)
    ax1.set_ylabel('Average Reward', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='tab:orange')

    ax1.plot(episode1, avg_reward1, color='tab:blue')
    ax1.plot(episode2, avg_reward2, color='tab:red')
    
    ax1.legend(['DQN', 'Double DQN'], prop = {'size' : 15})
    ax1.set_title('Biểu đồ reward trung bình tích lũy thu được qua các episode', fontsize=20)

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    if not os.path.isdir("./plots/"):
        os.mkdir("./plots/")
    plt.savefig("./plots/avg_reward.png", format='png')
    
    #plt.show()
    fig, ax2 = plt.subplots(figsize=(20, 10))
    ax2.set_xlabel('Episodes', fontsize=20)
    ax2.set_ylabel('Reward', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax2.plot(episode1, reward1, color='tab:blue')
    ax2.plot(episode2, reward2, color='tab:red')
    
    ax2.legend(['DQN', 'Double DQN'], prop = {'size' : 15})
    ax2.set_title('Biểu đồ reward thu được qua các episode', fontsize=20)
    fig.tight_layout()
    plt.savefig("./plots/reward.png", format='png')

if __name__ == '__main__':
    main()
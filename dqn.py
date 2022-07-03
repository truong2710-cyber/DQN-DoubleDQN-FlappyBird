import os
import glob
import random
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from game.flappy_bird import GameState
from model import NeuralNetwork, init_weights
from replay_buffer import ReplayBuffer
from utils import image_to_tensor, resize_and_bgr2gray

def train(q_model, start, args, hparams, target_model=None, mode='dqn'):
    assert mode in ['dqn', 'doubledqn']
    
    # define Adam optimizer
    optimizer = optim.Adam(q_model.parameters(), lr=hparams.lr)

    # initialize mean squared error loss
    criterion = nn.MSELoss()
    
    # initialize replay memory
    replay_buffer = ReplayBuffer(hparams.replay_memory_size)   ###

    # restore training
    if args.restore != None:
        q_model = torch.load(args.restore,
                             map_location='cpu' if not torch.cuda.is_available() else None)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            q_model = q_model.cuda()
        print("Restored training successfully!")

    if mode == 'doubledqn':
        # copy weights from q_model to target_model 
        target_model.load_state_dict(q_model.state_dict())

    # initialize reward logs
    reward_logs = []

    for episode in range(1, hparams.number_of_episodes+1):
        # instantiate game
        game_state = GameState()
        # initial action is do nothing
        action = torch.zeros([q_model.number_of_actions], dtype=torch.float32)
        action[0] = 1
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        # initialize epsilon value
        epsilon = hparams.initial_epsilon
        time_alive = 0
        total_reward = 0
        epsilon_decrements = np.linspace(hparams.initial_epsilon, hparams.final_epsilon, hparams.number_of_episodes)  ###

        # main infinite loop
        while True:
            # ACT TO GAIN EXPERIENCE
            # get output from the neural network
            output = q_model(state)[0]
            # initialize action
            action = torch.zeros([q_model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # epsilon-greedy exploration
            random_action = random.random() <= epsilon
            action_index = [torch.randint(q_model.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action
                            else torch.argmax(output)][0]

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()

            action[action_index] = 1

            # get next state and reward
            image_data_new, reward, terminal = game_state.frame_step(action)
            total_reward += reward
            image_data_new = resize_and_bgr2gray(image_data_new)
            image_data_new = image_to_tensor(image_data_new)
            state_new = torch.cat((state.squeeze(0)[1:, :, :], image_data_new)).unsqueeze(0)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
            # set state to be state_new
            state = state_new
            time_alive += 1
            # save transition to replay memory
            replay_buffer.append((state, action, reward, state_new, terminal))

            if terminal:
                break
            
        if episode > hparams.initial_observe_episode:
        # UPDATE Q-NETWORK 
        # sample random minibatch
            state_batch, action_batch, reward_batch, state_new_batch, terminal_batch = replay_buffer.sample(hparams.minibatch_size)
    
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_new_batch = state_new_batch.cuda()
                terminal_batch = terminal_batch.cuda()

            q_model.eval()
            
            if mode == 'doubledqn':
                target_model.eval()
            
            if mode == 'dqn':
                # get output for the next state
                action_new_batch = q_model(state_new_batch)
                # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                y_batch = torch.cat(tuple(reward_batch[i] if terminal_batch[i]
                                    else reward_batch[i] + hparams.gamma * torch.max(action_new_batch[i])
                                    for i in range(hparams.minibatch_size)))
            elif mode == 'doubledqn':
                # use q_model to evaluate action argmax_a' Q_current(s', a')_
                action_new = q_model.forward(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
                action_new_onehot = torch.zeros(hparams.minibatch_size, q_model.number_of_actions)
                action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).cuda()

                # use target_model to evaluate value
                # y = r + discount_factor * Q_tar(s', a')
                y_batch = (reward_batch + torch.mul(((target_model.forward(state_new_batch) *
                                        action_new_onehot).sum(dim=1) * terminal_batch),
                                        hparams.gamma))
            
            q_model.train()
            # extract Q-value
            q_value = torch.sum(q_model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.type(torch.float32)
            y_batch = y_batch.detach()

            # calculate loss
            loss = criterion(q_value, y_batch)

            # do backward pass
            loss.backward()
            optimizer.step()

            # epsilon annealing
            epsilon = epsilon_decrements[episode]

            # reward log for this episode
            reward_logs.extend([[episode, total_reward]])

            # save model and logs
            if episode % hparams.save_logs_freq == 0:
                reward_format = 'reward.npy' if mode == 'dqn' else 'reward_double.npy'
                np.save(os.path.join(args.logs_path, reward_format), np.array(reward_logs))
                model_list_double_dqn = glob.glob(os.path.join(args.checkpoint_path, '*double*.pth'))
                model_list_dqn = [file for file in glob.glob(os.path.join(args.checkpoint_path, '*.pth')) if file not in model_list_double_dqn]
                model_format = 'model_{}.pth' if mode == 'dqn' else 'model_double_{}.pth'
                # if maximum number of models is exceeded, remove the oldest model and save the current model
                if (len(model_list_dqn) >= hparams.maximum_model and mode == 'dqn') or (len(model_list_double_dqn) >= hparams.maximum_model and mode == 'doubledqn'):
                    min_step = min([int(li.split('\\')[-1][6:-4]) for li in model_list_dqn]) if mode == 'dqn' else \
                                min([int(li.split('\\')[-1][13:-4]) for li in model_list_double_dqn])
                    os.remove(os.path.join(args.checkpoint_path, model_format.format(min_step)))
                torch.save(q_model, os.path.join(args.checkpoint_path, model_format.format(episode)))
                print("Saved model to", os.path.join(args.checkpoint_path, model_format.format(episode)))

            # update target_model
            if mode == 'doubledqn' and episode % hparams.update_target_freq == 0:
                target_model.load_state_dict(q_model.state_dict())

        print("Episode:", episode, "time alive:", time_alive, "epsilon:", epsilon, "total reward:", total_reward)
    
    print("Finished training! Elapsed time:", time.time()-start)


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_new, reward, terminal = game_state.frame_step(action)
        image_data_new = resize_and_bgr2gray(image_data_new)
        image_data_new = image_to_tensor(image_data_new)
        state_new = torch.cat((state.squeeze(0)[1:, :, :], image_data_new)).unsqueeze(0)

        # set state to be state_new
        state = state_new
        
        #if terminal:
            #print("Game finished! Score:", game_state.getScore())

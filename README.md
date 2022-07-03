# DQN and Double DQN to play Flappy Bird in PyTorch

##### Dependencies:
* Python 3.9
* numpy
* torch
* opencv-python
* pygame


##### How to run:
* Run `pip3 install -r requirements.txt` to install dependencies.
* Run `python main.py --restore=./checkpoint/model_double_80000.pth` to test the pretrained model.
* Run `python dqn.py --train=True --mode='dqn'` to train the model from the beginning. You can also increase FPS in game/flappy_bird.py script for faster training.

##### Flags:
* `--restore`: load pretrained weight to restore training or test.
* `--logs_path`: path for reward log during training.
* `--checkpoint_path`: path for saving model checkpoints.
* `--train`: boolean, train or not, default=False.
* `--type`: `dqn` or `doubledqn`, default=`doubledqn`.
* `--hparam_path`: path of hyperparameter .json file.



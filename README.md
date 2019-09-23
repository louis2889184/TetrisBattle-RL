# TetrisBattle-RL

This is about training a RL agent to master once popular game -- Tetris Battle. Codes are forked from [1], which contains lots of famous RL algorithms. 

The challenge of mastering Tetris Battle: <br/>
1. It's hard to eliminate lines via random actions, so rewards are sparse. <br/>
2. The inputs are 600 * 800 images. (very big!) <br/>

Due to above challenges, the training is not successful by directly applying the algorithms in [1]. Therefore, I try to reduce the complexity of the games, and the concrete methods are<br/>
1. Use observations with smaller dimension (the hidden grid of the game). <br/> 
2. Reduce choices of the Tetriminos (which is blocks in Tetris).

## Requirements

- Python3<br/>
- [OpenAI-baseline](https://github.com/openai/baseline)<br>

other requirements: <br/>
`pip install -r requirements.txt`

## Usage

### Two kinds of Tetriminos
In this experiment, there are only I-shaped and O-shaped Tetriminos when playing. Under this setting, we can hugely reduce the complexity of the game and increase the probablity of the agent getting rewards.

To reproduce the results, follow the steps below,

1. Change `POSSIBLE_KEYS = ['I', 'O', 'J', 'L', 'Z', 'S', 'T']` in `TetrisBattle/settings.py` to `POSSIBLE_KEYS = ['I', 'I', 'I', 'I', 'O', 'O', 'O']`.
2. Train the RL agent with PPO algorithm: <br/>
`bash train_two_blocks.sh`.

This model uses raw screenshot as inputs and resized the input to 224 * 224.

## Experiments

### Two kinds of Tetriminos

Watch the performance of the learned agent: https://www.youtube.com/watch?v=vrmX3c4WIl0



## References
[1] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail <br/>
[2] https://github.com/louis2889184/TetrisBattle


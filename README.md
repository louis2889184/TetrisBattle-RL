# TetrisBattle-RL

This is about training a RL agent to master once popular game -- Tetris Battle. Codes are forked from [1], which contains lots of famous RL algorithms. 

The challenge of mastering Tetris Battle: <br/>
1. It's hard to eliminate lines via random actions, so rewards are sparse. <br/>
2. The inputs are 600 x 800 images. (very big!) <br/>

Due to above challenges, the training is not successful by directly applying the algorithms in [1]. Therefore, I try to reduce the complexity of the games, and the concrete methods are
1. Use observations with smaller dimension (the hidden grid of the game). <br/> 
2. Reduce choices of the blocks.

## Usage

1. Clone the Tetris Battle:

`https://github.com/louis2889184/TetrisBattle.git`

2. TBD...


## References
[1] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


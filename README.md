# AI-snake

The purpose of this code is to explore Reinforcement Learning through a fun game of snake. The AI tries to learn to play snake, with the idea of achieving a perfect score.

The snake is able to see the whole board, as well as the food as it moves, and a Q-learning algorithm is used to train it to recognize the goal of the game.

While you can download and run the code as it is now, it's still undergoing improvements. But if you want to play with it, the main and easiest parameters you can change and tune are the rows (size of the board), the reward and punishment values (for the Q-learning) passed to the ``Snake`` class in the ``generate_episode`` function, the model ``learning_rate``, ``rho`` and ``epsilon`` for the ``snakeAI`` function, as well as the epochs. 

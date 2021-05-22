import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from scipy.linalg import hankel
import matplotlib.pyplot as plt

from snake import Snake

#%%

def snakeAI(rows, hidden_dense_1, hidden_dense_2):
    # reshape the board and apple into 1 vector
    input_frame = Input(shape = (rows**2 + 2,)) 
    x = Dense(hidden_dense_1, activation = 'relu')(input_frame)
    x = Dense(hidden_dense_2, activation = 'relu')(x)
    # 4 states for movement
    outp = Dense(4, activation = 'softmax')(x) 
    
    snake_model = Model(input_frame, outp)
    snake_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.01, 
                                                               rho=0.95, 
                                                               epsilon=1e-07), loss="mse")
    # print(snake_model.summary())
    return snake_model
    

# create function to make NN input from game output
def prep_frame(apple, board): 
    inp = np.concatenate((np.asarray(apple),board.flatten()), axis = 0)
    inp = np.array(inp).reshape(-1,np.shape(inp)[0])
    return inp


def generate_episode(rows, model):
    # create snake object
    sn = Snake(rows, move_reward=1, apple_reward=5, punishment=-10)
    episode_history= []
    reward = 0
    alive = True
    while alive:
        # amalgamate board and apple locs
        state = prep_frame(sn.apple_xy,sn.board)
        # get a predictions from the model
        values = model.predict(state) 
        # choose one with biggest probability
        action = np.argmax(values) 
        # append needed info for the model to train
        episode_history.append((state,action,reward)) 
        # move the snake
        reward,alive = sn.move(action) 
    
    # once snake is dead, record last reward
    episode_history.append((None,None,reward)) 
    return np.array(episode_history)


def generate_state_action_values(episode_history, gamma):
    # all the rewards from the episode
    rewards = episode_history[:,2] 
    # num of future steps, counted from first frame
    frame_steps_num = len(episode_history) - 1 
    # create an array with exponents of gamma
    discount_factors = [gamma] * (frame_steps_num) 
    # raise gamma to powers from 0 - frame_steps_num
    discount_factors = np.array([x**i for i,x in enumerate(discount_factors)]) 
    # matrix with the future rewards where each row corresponds to next step, using a Hankel matrix
    future_rewards = hankel(rewards[1:]) 
    action_values = np.matmul(future_rewards,discount_factors)
    
    # action-values for each of the 4 movements possible
    values_matrix = np.zeros((len(action_values),4)) 
    for q in range(len(action_values)):
        # get action
        action = episode_history[q][1] 
        # for each frame action, get the action value
        value = action_values[q] 
        # for the particular action, populate the values
        values_matrix[q][action] = value 
    
    state_list = []
    for q in episode_history:
        # get board and snake states
        state_list.append(q[0]) 
    
    # concatenate in single matrix (rows = num of frames, cols = max number of moves)
    state_matrix = np.concatenate( state_list[:-1], axis=0 ) 
        
    return state_matrix, values_matrix

#%%
rows = 4

model = snakeAI(rows, hidden_dense_1 = 10, hidden_dense_2 = 8)
memory = [np.zeros((0,rows**2 + 2)),np.zeros((0,4))]

history_lenths = []
history_scores = []

num_ephochs = 3
# number of episodes after which the model gets updated
num_episodes = 3 
# hyperparameter -> after 1000 moves in the memory, moves will be randomly removed and 
# the model will be trained on the remaining 1000 moves
max_memory = 1000

for q in range(num_ephochs):
    print(" ============")
    print(" GENERATION ",q+1)
    print(" ============")
    # teach the model after every 3 episodes
    for episode in range(num_episodes): 
        # get the episode history
        history = generate_episode(rows, model) 
        # save frame length from episode
        history_lenths.append(len(history)) 
        # save total episode scores
        history_scores.append(sum(history[:,2])) 
        # get states and action-values
        states,action_values = generate_state_action_values(history,gamma=0.9) 
        # for each new episode add new states to first memory matrix
        memory[0] = np.concatenate((memory[0],states)) 
        # for each new episode add new action-values to second memory matrix
        memory[1] = np.concatenate((memory[1],action_values)) 
    
    if len(memory[0])>max_memory:
        # number of moves that need to be removed
        num_eps_del = len(memory[0]) - max_memory 
        # randomly choose N-number of moves to remove 
        eps_del = sorted(np.random.randint(0, max_memory, size=(num_eps_del))) 
        memory[0] = np.delete(memory[0], eps_del, axis = 0)
        memory[1] = np.delete(memory[1], eps_del, axis = 0)
    
     # train model on all 3 episodes
    model.fit(memory[0],memory[1])


plt.figure()
plt.plot(history_lenths)
plt.show()

plt.figure()
plt.plot(history_scores)
plt.show()




    
    



















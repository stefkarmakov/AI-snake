import numpy as np
import random
import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from scipy.linalg import hankel
import matplotlib.pyplot as plt

from snake import Snake

#%%

def snakeAI(rows, hidden_dense_1, hidden_dense_2):
    input_frame = Input(shape = (rows**2 + 2,)) # reshape the board and apple into 1 vector
    x = Dense(hidden_dense_1, activation = 'relu')(input_frame)
    x = Dense(hidden_dense_2, activation = 'relu')(x)
    outp = Dense(4, activation = 'softmax')(x) # 4 states for movement
    
    snake_model = Model(input_frame, outp)
    snake_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, 
                                                               rho=0.95, 
                                                               epsilon=1e-07), loss="mse")
    # print(snake_model.summary())
    return snake_model
    

def prep_frame(apple, board): # create function to make NN input from game output
    inp = np.concatenate((np.asarray(apple),board.flatten()), axis = 0)
    inp = np.array(inp).reshape(-1,np.shape(inp)[0])
    return inp


def generate_episode(rows, model):
    sn = Snake(rows, move_reward=1, apple_reward=5, punishment=-10) # create snake object
    episode_history= []
    reward = 0
    alive = True
    while alive:
        state = prep_frame(sn.apple_xy,sn.board) # amalgamate board and apple locs
        values = model.predict(state) # get a predictions from the model
        action = np.argmax(values) # choose one with biggest probability
        episode_history.append((state,action,reward)) # append needed info for the model to train
        reward,alive = sn.move(action) # move the snake
    
    episode_history.append((None,None,reward)) # once snake is dead, record last reward
    return np.array(episode_history)


def generate_state_action_values(episode_history, gamma):
    rewards = episode_history[:,2] # all the rewards from the episode
    frame_steps_num = len(episode_history) - 1 # num of future steps, counted from first frame
    discount_factors = [gamma] * (frame_steps_num) # create an array with exponents of gamma
    discount_factors = np.array([x**i for i,x in enumerate(discount_factors)]) # raise gamma to powers from 0 - frame_steps_num
    future_rewards = hankel(rewards[1:]) # matrix with the future rewards where each row corresponds to next step, using a Hankel matrix
    action_values = np.matmul(future_rewards,discount_factors)
    
    values_matrix = np.zeros((len(action_values),4)) # action-values for each of the 4 movements possible
    for q in range(len(action_values)):
        action = episode_history[q][1] # get action
        value = action_values[q] # for each frame action, get the action value
        values_matrix[q][action] = value # for the particular action, populate the values
    
    state_list = []
    for q in episode_history:
        state_list.append(q[0]) # get board and snake states
    
    state_matrix = np.concatenate( state_list[:-1], axis=0 ) # concatenate in single matrix (rows = num of frames, cols = max number of moves)
        
    return state_matrix, values_matrix

#%%
rows = 4

model = snakeAI(rows, hidden_dense_1 = 10, hidden_dense_2 = 8)
memory = [np.zeros((0,rows**2 + 2)),np.zeros((0,4))]

history_lenths = []
history_scores = []
for q in range(3000):
    print(" ============")
    print(" GENERATION ",q+1)
    print(" ============")
    for episode in range(3): # teach the model after every 3 episodes
        history = generate_episode(rows, model) # get the episode history
        history_lenths.append(len(history)) # save frame length from episode
        history_scores.append(sum(history[:,2])) # save total episode scores
        states,action_values = generate_state_action_values(history,gamma=0.9) # get states and action-values
        memory[0] = np.concatenate((memory[0],states)) # for each new episode add new states to first memory matrix
        memory[1] = np.concatenate((memory[1],action_values)) # for each new episode add new action-values to second memory matrix
    

    model.fit(memory[0],memory[1]) # train model on all 3 episodes

plt.figure()
plt.plot(history_lenths)
plt.show()

plt.figure()
plt.plot(history_scores)
plt.show()




    
    



















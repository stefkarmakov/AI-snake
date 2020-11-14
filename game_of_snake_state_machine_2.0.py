import numpy as np
import random
import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import logging
import matplotlib.pyplot as plt



from snake import Snake,Directions

def snakeAI(rows, hidden_dense_1, hidden_dense_2):
    input_frame = Input(shape = (rows**2 + 2,)) # reshape the board and apple into 1 vector
    x = Dense(hidden_dense_1, activation = 'relu')(input_frame)
    x = Dense(hidden_dense_2, activation = 'relu')(x)
    outp = Dense(4, activation = 'softmax')(x) # 5 states for movement
    
    snake_model = Model(input_frame, outp)
    # print(snake_model.summary())
    snake_model.compile(optimizer=tf.keras.optimizers.Adadelta(
         learning_rate=0.001, rho=0.95, epsilon=1e-07), loss="mse")
    return snake_model

def prep_frame(apple, board): # create function to make NN input from game output
    inp = np.concatenate((np.asarray(apple),board.flatten()), axis = 0)
    inp = np.array(inp).reshape(-1,np.shape(inp)[0])
    return inp



def generate_episode(model):
    board = Snake(rows=3,move_reward=1, apple_reward=10, punishment=0)
    episode_history= []
    reward = 0
    alive = True
    while alive:
        state = prep_frame(board.apple_xy,board.board)
        values = model.predict(state)
        action = np.argmax(values) + 1
        episode_history.append((state,action,reward))
        reward,alive = board.move(Directions(action))
    
    episode_history.append((None,None,reward))
    return episode_history


def generate_state_action_values(episode_history, gamma=0.95):
    action_values = [0.0] * (len(episode_history)-1)
    for q in range(1,len(episode_history)+1):
        frame = episode_history[-q]
        reward =frame[2]

        for m in range(len(episode_history) - q):
            action_distance = len(action_values) - q - m 
            action_values[m] += reward* (gamma**(action_distance))
    
    values_matrix = np.zeros((len(action_values),4))
    for q in range(len(action_values)):
        action = episode_history[q][1]
        value = action_values[q]
        values_matrix[q][action-1] = value 
    
    state_list = []
    for q in episode_history:
        state_list.append(q[0])
    
    state_matrix = np.concatenate( state_list[:-1], axis=0 )
    
    return state_matrix, values_matrix



model = snakeAI(3,3,3)
memory = [np.zeros((0,11)),np.zeros((0,4))]

history_lens = []
for q in range(500):
    print(" ============")
    print(" GENERATION ",q+1)
    print(" ============")
    for episode in range(3):
        history = generate_episode(model)
        history_lens.append(len(history))
        states,action_values = generate_state_action_values(history,gamma=0.98)
        memory[0] = np.concatenate((memory[0],states))
        memory[1] = np.concatenate((memory[1],action_values))
    

    model.fit(memory[0],memory[1])


plt.plot(history_lens)
plt.show()





















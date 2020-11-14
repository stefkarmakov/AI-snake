import numpy as np
import random
import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model



def snakeAI(rows, hidden_dense_1, hidden_dense_2):
    input_frame = Input(shape = (rows**2 + 2,)) # reshape the board and apple into 1 vector
    x = Dense(hidden_dense_1, activation = 'relu')(input_frame)
    x = Dense(hidden_dense_2, activation = 'relu')(x)
    outp = Dense(5, activation = 'softmax')(x) # 5 states for movement
    
    snake_model = Model(input_frame, outp)
    # print(snake_model.summary())
    return snake_model

def prep_frame(apple, board): # create function to make NN input from game output
    inp = np.concatenate((np.asarray(apple),board.flatten()), axis = 0)
    inp = np.array(inp).reshape(-1,np.shape(inp)[0])
    return inp



rows = 8
apple_reward = 5
move_reward = 1
punishment = -100
hidden_dense_1 = 40
hidden_dense_2 = 20
archive_max_size = 2**8 # max size of frames the snake will learn on

board_history = []
apple_history = []
score_history = [0] # start game with 0 score
prediction = []
new_frames = []

sn = snake(rows, move_reward, apple_reward, punishment) # create snake object
apple_history.append(sn.apple_xy) # first apple loc
board_history.append(sn.board) # first board 
snake_model = snakeAI(rows, hidden_dense_1, hidden_dense_2) # create NN model
# sn.turns
# sn.score

init_frame = prep_frame(apple_history[0], board_history[0]) # create the initial frame from apple and board
new_frames.append(init_frame)
prediction.append(snake_model.predict(new_frames[0]))

in_progress = True
archive_frames = []
archive_score = []

while in_progress:
    last_pred = prediction[-1][0] # take array of last predicted probability distribution
    model_move = np.where(last_pred == max(last_pred))[0][0] # find max value idx // [0][0] is to get a value from the array
    current_board, in_progress = sn.move(model_move) # apply move and update game
    
    board_history.append(current_board) # add new board
    apple_history.append(sn.apple_xy) # add apple position, whether it has changed or not
    score_history.append(sn.score)
    new_frames.append(prep_frame(apple_history[-1], board_history[-1])) # create new frame
    prediction.append(snake_model.predict(new_frames[-1])) # add to predictions


archive_frames.append(new_frames) # add new frames to learning pool
archive_score.append(score_history) # add new scores to learning pool
if len(archive_frames) > archive_max_size:
    del_frames = random.sample(range(len(archive_frames)), len(archive_frames) - archive_max_size) # remove random frames so the archive stays same size
    for index in sorted(del_frames, reverse=True): # del in reverse order so that the indices don't get overwritten
        del archive_frames[index] # delete all the excess frames 
        del archive_score[index] # delete all the excess scores 









    
    



















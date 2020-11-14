import numpy as np
import random
import operator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


class snake():
    def __init__(self, rows, move_reward, apple_reward, punishment):
    
        self.body = []
        self.turns = []
        self.board = np.zeros((rows,rows)) # create board from all 0s
        self.head = int(rows/2), int(rows/2) # the head of the snake is at middle
        self.body.append(self.head) # add head to body
        self.board[self.head] = 1 # the head of the snake is at middle
        self.apple_xy = self.random_apple() # create an apple
        self.score = 0
        self.apple_reward = apple_reward
        self.move_reward = move_reward
        self.punishment = punishment
        self.no_score_count = 0 # count the number of moves without finding/eating an apple
        
    # 0 - continue  ||  1 - left  ||  2 - right  ||  3 - up  ||  4 - down  
      
    def move(self,key): 
        self.score += self.move_reward # for every move, reward the snake
        if key == 0:
            if not(self.turns): # if it's the first movement entry
                self.turns.append((0,0)) # just don't move
            else:
                self.turns.append(self.turns[-1]) # take previous movement and continue with it
        elif key == 1:
            self.turns.append((0,-1))
        elif key == 2:
            self.turns.append((0,1))
        elif key == 3:
            self.turns.append((-1,0))
        elif key == 4:
            self.turns.append((1,0))
        
        for part in range(len(self.body)): # update the body so that each part turns the way it should
            # the head reaction is instantaneous, but the other parts have a delay based on how far they are 
            # from the head -> go back n turns for n-th body part 
            self.body[part] = tuple(map(operator.add, self.body[part], self.turns[-1-part]))
        
        is_game_finished = self.end_game() # after every move, check if snake bit itself or hit the wall
        if is_game_finished == True: # if process/game has finished
            return self.board, False
        else:
            self.eat_apple() # after every move, check if snake ate apple
            self.redraw_board()# after every move, redraw board
            return self.board, True
    
    def random_apple(self):
    
        while True: # create apple at a random location
            apple_x = random.randrange(rows)
            apple_y = random.randrange(rows)
            if (apple_x,apple_y) in self.body: # check if on that location there is a snake body
                continue # if yes, continue trying
            else: 
                break # if no, choose that position
     
        return apple_x, apple_y
    
                    
    def eat_apple(self):   
        
        if self.body[0] == self.apple_xy: # if head is on top of apple
            tail_pos = self.body[-1] # find position of tail
            tail_move = self.turns[-len(self.body)] # take the turn that the tail is doing
            self.body.append(tuple(map(operator.sub, tail_pos, tail_move))) # add one more part to body which will move in tail dir
            self.apple_xy = self.random_apple() # get coordinates of new random apple
            self.score += self.apple_reward # add reward for eaten apple
            self.no_score_count = 0 # resent the movement count
        else: # if not, add 1 to the movement count
            self.no_score_count += 1
    
    
    def redraw_board(self): 
        
        self.board = np.zeros((rows,rows)) # redraw the empty board and then place the new body position
        r, c = zip(*self.body)
        self.board[r, c] = np.ones(len(self.body)) # redraw the snake on the board based on the new body positions
        
        
    def end_game(self):
        
        is_game_finished = False
        if (
            self.body[0] in self.body[1:] or # check if the head has the same coordinates as the rest of the body
            [a>=0 for a in self.body[0]] != [True,True] or # check if snake has hit the top or left wall
            [b<=rows-1 for b in self.body[0]] != [True,True] or # check if snake has hit the bottom or right wall
            self.no_score_count >= 2*(rows**2) # if the snake hasn't eaten an apple in these many turns, end game
            ): 
                self.score += self.punishment # add punishment
                is_game_finished = True # 
                
        return is_game_finished


#%%

def snakeAI(rows, hidden_dense_1, hidden_dense_2):
    input_frame = Input(shape = (rows**2 + 2,)) # reshape the board and apple into 1 vector
    x = Dense(hidden_dense_1, activation = 'relu')(input_frame)
    x = Dense(hidden_dense_2, activation = 'relu')(x)
    outp = Dense(5, activation = 'softmax')(x) # 5 states for movement
    
    snake_model = Model(input_frame, outp)
    # print(snake_model.summary())
    return snake_model
    
#%%

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









    
    



















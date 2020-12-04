import numpy as np
import random
import operator


class Snake():
    def __init__(self, rows, move_reward, apple_reward, punishment):
        
        self.body = []
        self.turns = []
        self.board = np.zeros((rows,rows)) # create board from all 0s
        self.head = int(rows/2), int(rows/2) # the head of the snake is at middle
        self.body.append(self.head) # add head to body
        self.board[self.head] = 1 # the head of the snake is at middle
        self.apple_xy = self.random_apple(rows) # create an apple
        self.apple_reward = apple_reward
        self.move_reward = move_reward
        self.punishment = punishment
        self.no_score_count = 0 # count the number of moves without finding/eating an apple
        
    # 0 - left  ||  1 - right  ||  2 - up  ||  3 - down  
      
    def move(self,key):
        rows = len(self.board) # define rows again, instead of making it global
        self.step_reward = 0
        self.step_reward += self.move_reward # for every move, reward the snake
#         if key == 0:
#             if not(self.turns): # if it's the first movement entry
#                 self.turns.append((0,0)) # just don't move
#             else:
#                 self.turns.append(self.turns[-1]) # take previous movement and continue with it
        if key == 0:
            self.turns.append((0,-1))
        elif key == 1:
            self.turns.append((0,1))
        elif key == 2:
            self.turns.append((-1,0))
        elif key == 3:
            self.turns.append((1,0))
        
        for part in range(len(self.body)): # update the body so that each part turns the way it should
            # the head reaction is instantaneous, but the other parts have a delay based on how far they are 
            # from the head -> go back n turns for n-th body part 
            self.body[part] = tuple(map(operator.add, self.body[part], self.turns[-1-part]))
        
        is_game_finished = self.end_game(rows) # after every move, check if snake bit itself or hit the wall
        if is_game_finished == True: # if process/game has finished
            return self.step_reward, False
        else:
            self.eat_apple(rows) # after every move, check if snake ate apple
            return self.step_reward, True
    
    def random_apple(self, rows):
        free_locs = np.where(self.board.ravel() == 0)[0] # find all the board places where the body isn't
        apple_loc = random.choice(free_locs) # choose random free location
        apple_x = apple_loc // rows # find the row of the location
        apple_y = apple_loc % rows # find the column of the location
       
        return apple_x, apple_y
    
                    
    def eat_apple(self, rows):   
        
        if self.body[0] == self.apple_xy: # if head is on top of apple
            tail_pos = self.body[-1] # find position of tail
            tail_move = self.turns[-len(self.body)] # take the turn that the tail is doing
            self.body.append(tuple(map(operator.sub, tail_pos, tail_move))) # add one more part to body which will move in tail dir
            self.redraw_board(rows) # redraw board after the body has increased, but before the generation of a new apple
            self.apple_xy = self.random_apple(rows) # get coordinates of new random apple
            self.step_reward += self.apple_reward # add reward for eaten apple
            self.no_score_count = 0 # resent the movement count
        else: # if not, redraw board and add 1 to the movement count
            self.redraw_board(rows)
            self.no_score_count += 1
    
    
    def redraw_board(self, rows): 
        
        self.board = np.zeros((rows,rows)) # redraw the empty board and then place the new body position
        r, c = zip(*self.body)
        self.board[r, c] = np.ones(len(self.body)) # redraw the snake on the board based on the new body positions
        
        
    def end_game(self, rows):
        
        is_game_finished = False
        if (
            self.body[0] in self.body[1:] or # check if the head has the same coordinates as the rest of the body
            [a>=0 for a in self.body[0]] != [True,True] or # check if snake has hit the top or left wall
            [b<=rows-1 for b in self.body[0]] != [True,True] or # check if snake has hit the bottom or right wall
            self.no_score_count >= 2*(rows**2) # if the snake hasn't eaten an apple in these many turns, end game
            ): 
                self.step_reward += self.punishment # add punishment
                is_game_finished = True 
                
        return is_game_finished
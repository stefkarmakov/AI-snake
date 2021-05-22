import numpy as np
import random
import operator


class Snake():
    def __init__(self, rows, move_reward, apple_reward, punishment):
        
        self.body = []
        self.turns = []
        # create board from all 0s
        self.board = np.zeros((rows,rows)) 
        # the head of the snake is at middle
        self.head = int(rows/2), int(rows/2)
        # add head to body
        self.body.append(self.head) 
        # the head of the snake is at middle
        self.board[self.head] = 1 
        # create an apple
        self.apple_xy = self.random_apple(rows) 
        self.apple_reward = apple_reward
        self.move_reward = move_reward
        self.punishment = punishment
        # count the number of moves without finding/eating an apple
        self.no_score_count = 0 
        
    # 0 - left  ||  1 - right  ||  2 - up  ||  3 - down  
      
    def move(self,key):
        # define rows again, instead of making it global
        rows = len(self.board) 
        self.step_reward = 0
        # for every move, reward the snake
        self.step_reward += self.move_reward 
        if key == 0:
            self.turns.append((0,-1))
        elif key == 1:
            self.turns.append((0,1))
        elif key == 2:
            self.turns.append((-1,0))
        elif key == 3:
            self.turns.append((1,0))
            
        # update the body so that each part turns the way it should the head 
        # reaction is instantaneous, but the other parts have a delay based on 
        # how far they are from the head -> go back n turns for n-th body part 
        for part in range(len(self.body)): 
            self.body[part] = tuple(map(operator.add, self.body[part], self.turns[-1-part]))
        
        # after every move, check if snake bit itself or hit the wall
        is_game_finished = self.end_game(rows)
        # if process/game has finished
        if is_game_finished == True: 
            return self.step_reward, False
        else:
            # after every move, check if snake ate apple
            self.eat_apple(rows) 
            return self.step_reward, True
    
    
    def random_apple(self, rows):
        # find all the board places where the body isn't
        free_locs = np.where(self.board.ravel() == 0)[0] 
        # choose random free location
        apple_loc = random.choice(free_locs) 
        # find the row of the location
        apple_x = apple_loc // rows 
        # find the column of the location
        apple_y = apple_loc % rows 
       
        return apple_x, apple_y
    
                    
    def eat_apple(self, rows):   
        
        # if head is on top of apple
        if self.body[0] == self.apple_xy: 
            # find position of tail
            tail_pos = self.body[-1] 
            # take the turn that the tail is doing
            tail_move = self.turns[-len(self.body)] 
            # add one more part to body which will move in tail dir
            self.body.append(tuple(map(operator.sub, tail_pos, tail_move))) 
            # redraw board after the body has increased, but before the generation of a new apple
            self.redraw_board(rows) 
            # get coordinates of new random apple
            self.apple_xy = self.random_apple(rows) 
            # add reward for eaten apple
            self.step_reward += self.apple_reward 
            # resent the movement count
            self.no_score_count = 0 
        # if not, redraw board and add 1 to the movement count
        else: 
            self.redraw_board(rows)
            self.no_score_count += 1
    
    
    def redraw_board(self, rows): 
        
        # redraw the empty board and then place the new body position
        self.board = np.zeros((rows,rows)) 
        r, c = zip(*self.body)
        # redraw the snake on the board based on the new body positions
        self.board[r, c] = np.ones(len(self.body)) 
        
        
    def end_game(self, rows):
        
        is_game_finished = False
        if (
            # check if the head has the same coordinates as the rest of the body
            self.body[0] in self.body[1:] or 
            # check if snake has hit the top or left wall
            [a>=0 for a in self.body[0]] != [True,True] or 
            # check if snake has hit the bottom or right wall
            [b<=rows-1 for b in self.body[0]] != [True,True] or 
            # if the snake hasn't eaten an apple in these many turns, end game
            self.no_score_count >= 2*(rows**2) 
            ): 
                # add punishment
                self.step_reward += self.punishment 
                is_game_finished = True 
                
        return is_game_finished
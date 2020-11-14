
import numpy as np
import random
import operator
from enum import Enum

class Directions(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class Snake():
    def __init__(self, rows=10, move_reward=1, apple_reward=10, punishment=1):
    
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
        
    
    def __str__(self):
        printable = np.copy(self.board)
        printable[self.apple_xy[0]][self.apple_xy[1]] = 3
        return str(printable)
            
    # 0 - continue  ||  1 - left  ||  2 - right  ||  3 - up  ||  4 - down  
      
    def move(self,key): 
        self.score += self.move_reward # for every move, reward the snake
        if key == Directions.NONE:
            if not(self.turns): # if it's the first movement entry
                self.turns.append((0,0)) # just don't move
            else:
                self.turns.append(self.turns[-1]) # take previous movement and continue with it
        elif key == Directions.LEFT:
            self.turns.append((0,-1))
        elif key == Directions.RIGHT:
            self.turns.append((0,1))
        elif key == Directions.UP:
            self.turns.append((-1,0))
        elif key == Directions.DOWN:
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
            apple_x = random.randrange(len(self.board))
            apple_y = random.randrange(len(self.board))
            if self.board[apple_x,apple_y] == 1: # check if on that location there is a snake body
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
        
        self.board = np.zeros((len(self.board),len(self.board))) # redraw the empty board and then place the new body position
        r, c = zip(*self.body)
        self.board[r, c] = np.ones(len(self.body)) # redraw the snake on the board based on the new body positions
        
        
    def end_game(self):
        
        is_game_finished = False
        if (
            self.body[0] in self.body[1:] or # check if the head has the same coordinates as the rest of the body
            [a>=0 for a in self.body[0]] != [True,True] or # check if snake has hit the top or left wall
            [b<=len(self.board)-1 for b in self.body[0]] != [True,True] or # check if snake has hit the bottom or right wall
            self.no_score_count >= 2*(len(self.board)**2) # if the snake hasn't eaten an apple in these many turns, end game
            ): 
                self.score += self.punishment # add punishment
                is_game_finished = True # 
                
        return is_game_finished




if __name__ == "__main__":
    s = Snake(rows=3,move_reward=2,apple_reward=10,punishment=1)
    print(s)
    while True:
        key = input("Input: ")
        if key == "exit":
            break
        _,alive = s.move(Directions(int(key)))
        if not alive:
            print("Dead!")
            break
        print(s)

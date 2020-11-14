
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
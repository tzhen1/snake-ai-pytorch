import pygame
import random # for food
from enum import Enum
from collections import namedtuple # assigns meaning to positions in a tuple = readable + lighter
import numpy as np
pygame.init() # init all modules

#MACROS
BLOCK_SIZE = 20 # constant for 1 block on screen
SPEED = 40 # speed of snake

#RGB tuple
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

#pygame fonts
font1 = pygame.font.Font('arial.ttf', 25)

Point = namedtuple('Point', 'x,y') #'Point' is name of tuple(lighter), Point(x,y) access with Point.x, access the named tuple. 
# takes in a string, but variables seperated 

#PYTORCH
#reset game for agent

#reward, eat food = +10, gameover = -10, else = 0 

#play func takes action, outputs direction

#keep track game iteration

#is collision changes

#game
class Direction(Enum): #class inherit from enum
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI: # contains functions
    # define init function, needs to be __init__ due to __name_
    def __init__(self, w = 640, h = 480): # display itself, default pixels width , height
        self.w = w
        self.h = h

        #init display, self.any is a new variable
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snakey')
        self.clock = pygame.time.Clock() # speed of game
        self.Reset() # init/reset game
        
    # AI, refactor init game state for resetting when agent finish a game
    def Reset(self):
        self.direction = Direction.RIGHT # snake initial direction to right

        #head of snake, starts middle of display
        self.head = Point(self.w / 2, self.h / 2) #stores the cords in a tuple named 'head' which can be accessed in next code
        
        #create snake body of 3 body size, list contains info of 
        self.snake = [  self.head, 
                        Point(self.head.x - BLOCK_SIZE, self.head.y), #head, snake set point abit left of the head of snake (mid), same y
                        Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]  # set 3rd part of body 2 times left of head

        self.score = 0
        self.food = None
        self.Place_Food() 
        self.frame_iteration = 0 # makes first iteration = 0


    def Place_Food(self): # gets only self, placed randomly inside our dimensions
        food_x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE ) * BLOCK_SIZE # x=random int from 0 and [width - blocksize] (so food is in block)
        food_y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE ) * BLOCK_SIZE
    #in order to get only block size multiples so food inside a block use trick:the width/height divide by blocksize then * by blocksize get 
    #back a random int of a multiple of only block sizes
        self.food =  Point(food_x,food_y) #sets food position from this function in a tuple, access by self.food.x or y
        if self.food in self.snake: #dont place food in snake, checks if in self.snake list with 3 cords on the body
            self.Place_Food() # new random cords

    def Play_Step(self, action): # play step needs to get an action from agent
        # fram iterates after every step
        self.frame_iteration += 1

        #1 collect user input
        for event in pygame.event.get(): # gets all user events
            if event.type == pygame.QUIT:
                pygame.quit()
                quit() # python prog
   
        #2 move snake 
        self.Move(action) # updates new snake head position, calls self.move with the action
        self.snake.insert(0, self.head) #transfer new position into the self.snake list, at index 0 = head
        # this inserts a new block at front of the head, when we use pop below to remove it at the end of snake keeps its size

        #3 check gameover, boundary or into itself
        reward = 0 # init
        game_over = False
        #check if collide or if nothing happens after awhile (no food or die) we will break
        if self.Is_collision() or self.frame_iteration > 100*len(self.snake): #longer than 100 x snake body (list, each var is a body part) 
         # the longer the snake = more time pass, but if gets larger than 100* snake that doesnt change body size (same list size), its stalling
            game_over = True
            reward = -10
            return reward, game_over, self.score 
        
        #4 place new food (when got) or just moving (when move 1 block, we created new one, so just remove last one by pop)
        if self.head == self.food: # if collide
            self.score += 1
            reward = 10
            self.Place_Food()
        else:
            self.snake.pop() # removes last element of snake
        
        #5 update ui (snake + background) + clock
        self.Update_UI() 
        self.clock.tick(SPEED) # how fast frame updates

        #6 return gameover + score
        return reward, game_over, self.score  

    def Is_collision(self, pt = None): # ai we will use this to calculate danger too, set point default = none
        #if nothing, then sets point where head is
        if pt is None:
            pt = self.head
        #hits boundary
        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0 # hit left or right width wise
        or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True # yes collision
        #hits itself (rest of list not head at index 0)
        if pt in self.snake[1:]: #if head in other body parts so not index 0 
            return True
        return False # no collision    

    def Update_UI(self):
        self.display.fill(BLACK) #fill screen with black

        #draw snake
        for pt in self.snake: #iterate over all points of snake body (points of body in its list)
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))#draw on display, blue, rectangle snake
            #set position of rect to self.snake tuple's x + y cords, size of rect (w x h)
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)) # draw innards of snake at same postion of body

        #draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        #score text
        text_score = font1.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text_score, [0,0]) # upper left
        pygame.display.flip() # updates full display

    def Move(self, action): # gets action, use this to get new direction 
        #(straight/left/right), turning depending on current direction (clockwise/anticlk)
        Clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] #anti-clk is just negative
        
        #get current direction in index form from clock wise list
        curr_dir_idx = Clock_wise.index(self.direction) #self direction is one of the ENUM values, make curr_dir = to a direction in list

        if np.array_equal(action, [1,0,0]): # if action is = o array (straight dir) 
            new_dir = Clock_wise[curr_dir_idx] # same direction, no change
        if np.array_equal(action, [0,1,0]): # right turn
            new_dir_idx = (curr_dir_idx + 1) % 4 #if val >4,if index 4(empty),4 % 4 (finds remainder) , back to index 0, if just 1%4, still 1 
            new_dir = Clock_wise[new_dir_idx] 
        else: # [0,0,1] left
            new_dir_idx = (curr_dir_idx - 1) % 4 # anti-clk
            new_dir = Clock_wise[new_dir_idx] 
        self.direction = new_dir

        x = self.head.x #get original x + y position
        y = self.head.y

        # checks new direction
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE # move right by a block
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE 
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE 
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE # pixel 0 at top, increases downwards
        
        self.head = Point(x,y) #head now a new pos



# NOT USED AS NO USER INPUT 
# if __name__ == '__main__': # if run as main process
#     game = SnakeGameAI() #create snake game 

#     #game loop endless
#     while True:
#         play_game_over, game_score = game.play_step() # executing func returns gameover + score 

#         if play_game_over == True:
#             break #exits loop to quit

#     print('Final score', game_score)
#     pygame.quit()


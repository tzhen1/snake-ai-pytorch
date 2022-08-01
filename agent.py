import torch
import random
import numpy as np
from collections import deque
from game_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE  # take class + tuple
from model import Linear_QNet, QTrainer
from helper import plot # import plot function

#IMPROVE AGENT, KEEPS TRAPPIGNG ITSELF, CONVOLUTIONAL NET?
#parameter see weather head sururounded by body

#MACROS
MAX_MEMORY = 100_000 # items
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        self.num_of_games = 0 
        self.epsilon = 0 # controls randomness
        self.gamma = 0.9 # discount rate (play around, but <1)
        self.memory = deque(maxlen = MAX_MEMORY) # if exceeds,popleft() [removes elements from the left]
        #instance of model + trainer
        self.model = Linear_QNet(11, 256, 3) #11 i/p (state val), hidden (play around with neurons num), o/p size (3 for action [1,0,0])
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma) # todo

        #todo: model,trainer

    def Get_State(self, game): # gets self + snake game, calculates state (11 variables)
        
        #get snake head, 1st on the list (self.snake in )
        snake_head = game.snake[0] 

        # 1 block around snake head (for check hits edges)
        left_of_head = Point(snake_head.x - BLOCK_SIZE, snake_head.y) # cords in a tuple
        right_of_head = Point(snake_head.x + BLOCK_SIZE, snake_head.y)
        up_of_head = Point(snake_head.x, snake_head.y - BLOCK_SIZE)
        down_of_head = Point(snake_head.x, snake_head.y + BLOCK_SIZE)
        

        #find current direction, '==' check if current game direction is equal to one of the Direction
        #only 1 of the dir will be true, rest false
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [ #11 values states
            # Dangers straight (same direction as block ahead), if u go any direction here, and there is collision on points around head
            (dir_r and game.Is_collision(right_of_head)) or #if right true(=1) & there is collision to the right of head = danger true
            (dir_l and game.Is_collision(left_of_head)) or #...
            (dir_u and game.Is_collision(up_of_head)) or 
            (dir_d and game.Is_collision(down_of_head)),

            # Dangers for right turn
            (dir_u and game.Is_collision(right_of_head)) or #if cur dir = up + there collision to right of head = true
            (dir_d and game.Is_collision(left_of_head)) or 
            (dir_l and game.Is_collision(up_of_head)) or 
            (dir_r and game.Is_collision(down_of_head)),

            # Danger left
            (dir_d and game.Is_collision(right_of_head)) or 
            (dir_u and game.Is_collision(left_of_head)) or 
            (dir_r and game.Is_collision(up_of_head)) or 
            (dir_l and game.Is_collision(down_of_head)),
            
            # Move direction, only 1 is true
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location in bool (multple values could be true) 
            game.food.x < game.head.x, # food left, if x-axis, food is left side that snake head
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
            ]
        
        return np.array(state, dtype = int) #puts state into array datatype of array = int, converts bool to 0 or 1

    def Remember(self, state, action, reward, next_state, gameover): #remembers reward from action, store next state
        #deque, append all info in the order
        self.memory.append((state, action, reward, next_state, gameover)) #use () as we want it as 1 tuple (as 1 element)
        #popleft if exceed max mem

    def Train_Long_Memory(self): # get 1000 samples (batch) from memory
        if len(self.memory) > BATCH_SIZE: # if too many elements
            # get a smaller random sample from mem (batch size, 1000 tuples)
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns a list of 1000 tuples, since xceeded 1000 in memory
        else: # not 1000 elements
            mini_sample = self.memory # use whole memory if just 1000 or less samples

        #train for multiple parameters (lists) by extract info from mini_sample, group states, actions etc
        states, actions, rewards, next_states, gameovers = zip(*mini_sample) #zip, puts every state together, every action together..
        self.trainer.Train_step(states, actions, rewards, next_states, gameovers) #multiples states

        #or can use for loop, iterates over mini sample
        # for self, states, actions, rewards, next_states, gameovers in mini_sample #states... in 1 mini sample
            # self.trainer.train_step(self, state, action, reward, next_state, gameover)
        

    #train for 1 parameter (list of variables)
    def Train_Short_Memory(self, state, action, reward, next_state, gameover): #train for 1 gamestep, uses the info
        self.trainer.Train_step(state, action, reward, next_state, gameover) # 1 state

    def Get_Action(self, state): #get action based on state
        # state with random move (tradeoff between exploring or exploiting after agent improves)
        self.epsilon = 80 - self.num_of_games #epsilon = randomness, as num of games rises, randomness falls
        final_move = [0,0,0] # start empty 0,0,0, one will be true

        if random.randint(0,200) < self.epsilon: #random number, less than epsilon, so we will do a random move
            move = random.randint(0,2) #gives 0,1 or 2 for the 3 indexes [0,0,0]
            final_move[move] = 1 # make one of values =1
            #as more games, epsilon decreases, less random moves
        else: # move based on model
            state_0 = torch.tensor(state, dtype = torch.float) #array in tensor, gets state as i/p, float data type

            # predict action based on 1 state, prediction = raw val, e.g [5.2,1,0.9] where we need take max value to get [1,0,0]
            prediction = self.model(state_0) #executes forward func in model as prediction
            move = torch.argmax(prediction).item() # item, converts the tensor to 1 int, so [1,0,0] -> [0] or [0,1,0] ->[1]
            final_move[move] = 1 # makes one of initial values 1, which is a move either (straight/l/r)

        return final_move #func returns this


        
#global func
def Train():
    plot_score = [] #empty start list
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent() # agent object, access to those variables
    game = SnakeGameAI() # snake game obj

    agent.model.Load()


    while True:
        #get old state
        state_old = agent.Get_State(game)

        #get move (action) based on current state
        final_move = agent.Get_Action(state_old)

        #perform move + get new state
        reward, gameover, score = game.Play_Step(final_move) # func play step gets action
        state_new = agent.Get_State(game) # new game

        #train short mem (1 step)
        agent.Train_Short_Memory(state_old, final_move, reward, state_new, gameover)

        #remember
        agent.Remember(state_old, final_move, reward, state_new, gameover)

        if gameover: # true
        #train long mem (experience replay), trains again on all previous moves + games, improves most , plot result
            game.Reset() # from snake_game.py
            agent.num_of_games += 1
            agent.Train_Long_Memory()

            if score > record:
                record = score
                agent.model.Save_Model_State() # model will save the record

            print('Game', agent.num_of_games, 'Score', score, 'Record:', record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score/ agent.num_of_games # get games from another func
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)



#start agent.py
if __name__ == '__main__':
    Train()
    #start using python agent.py

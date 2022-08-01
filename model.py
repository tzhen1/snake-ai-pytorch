import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module): #feed forward neural net, 3 layers, can be improved

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        #linear layers
        self.linear1 = nn.Linear(input_size, hidden_size) # i/p = i/p size, o/p = hidden size
        self.linear2 = nn.Linear(hidden_size, output_size) #i/p = hidden, o/p = o/p size

    def forward(self, x): # x is tensor
        
        # apply linear layers using actuation function, from functional module
        x = F.relu(self.linear1(x)) #1st linear layer then apply actuation func tensor x as i/p  
        #apply 2nd layer
        x = self.linear2(x) #dont needactutation func here, just use raw numbers 
        return x 

    def Save_Model_State(self, file_name = 'model.pth'): # save model, gets file name as i/p from user
        model_folder_path = './model' #new folder in current dir called model
        #check if file exists in the folder
        if not os.path.exists(model_folder_path): # if path doesnt exist
            os.makedirs(model_folder_path) #create folder

        file_name = os.path.join(model_folder_path, file_name) #join file to folder path

        #save this
        torch.save(self.state_dict(), file_name) #save state model as dictionary to the file
    
    def Load(self, file_name = 'model.pth'):
        model_folder_path = './model' 
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.exists(model_folder_path):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print ('Loading existing state dict.')
            return True

        print ('No existing state dict found. Starting from scratch.')
        return False


        
class QTrainer:

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        #pytorch optimisation, adam optimiser
        self.optimiser = optim.Adam(model.parameters(), lr = self.lr)

        #loss function,  loss = (Qnew - Q )^2 , just mean squared error
        self.criterion = nn.MSELoss()

    def Train_step(self, state, action, reward, next_state, gameover): 
        #tensor array
        state = torch.tensor(state, dtype = torch.float) # tensor for a variable and its data type
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        # no tesnor for gameover, not needed

        #handle multiple batches, so check length (matrix)
        if len(state.shape) == 1: # state matrix only 1 column, 1D, so only has 1 list of values, as only have 1 batch, (1, x) 
            # we want it be be shown as (1, x), 1 = batches, x = values for state

            #reshape this for multiple batches, if not then this if code will be skipped, already big batch n , (n,x) = correct
            state = torch.unsqueeze(state, 0) # unsqueeze adds a dimension to tensor (array), adding dimension at 0th position (at start)
            # (1) will become (1,1), 1 row, 1 column

            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameover = (gameover, ) #made into tuple with 1 value (using blank)

            
        #bellman equation
        #predicted q values with current state (state 0)
        pred = self.model(state) #is the action (3 values)

        target = pred.clone() # makes target a clone of prediction, this will be Q_new, prediction is Q old
        
        #iterate over the range of tensors 
        for idx in range(len(gameover)): # all values same size as gameover, e.g action reward etc
            Q_new = reward[idx] #reward of curr index

            if not gameover[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) # max value of next prediction

            target[idx][torch.argmax(action[idx]).item()] = Q_new #target of curr idx, 1 value not tensor, 
            #target was clone of prediction, now is q_new

        #steps for  Q_new
        # Q_new = r+ y * max(next_predicted Q value) # formaula for new Q, but outputs 1 value, do if not gameover
        # pred.clone() # to get it back to same format as pred before get 3 values e.g [1,0,0]
        # pred[argmax(action)] = Q_new # get max of action [1,0,0], gets 1, sets to Q_new

        self.optimiser.zero_grad() #empty gradient
        loss = self.criterion(target, pred) # tager = Q_new now, pred = Q
        loss.backward() # back propagation
        #update gradient
        self.optimiser.step()





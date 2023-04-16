import torch.nn as nn
import torch.nn.functional as F

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation='relu', init_type='uniform' , dropout = 0):
        super(MLP, self).__init__()
        self.units = units
        self.n_layers = len(units) # including input and output layers
        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid(),
                             'none': False}
        self.activation = valid_activations[hidden_layer_activation]
        self.dropout = dropout

        #####################################################################################
        # TODO: Implement the model architecture with respect to the units: list            #
        # use nn.Sequential() to stack layers in a for loop                                 #
        # It can be summarized as: ***[LINEAR -> ACTIVATION]*(L-1) -> LINEAR -> SOFTMAX***  #
        # Use nn.Linear() as fully connected layers                                         #
        #####################################################################################
        layers = []
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(units[i] , units[i+1]))
            if self.activation:
                layers.append(self.activation) 
            if self.dropout != 0:
                layers.append(nn.Dropout(self.dropout))
                
        self.mlp = nn.Sequential(*layers)       
                

        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################

    def forward(self, X):
        #####################################################################################
        # TODO: Forward propagate the input                                                 #
        # ~ 2 lines of code#
        # First propagate the input and then apply a softmax layer                          #
        #####################################################################################
        output = self.mlp(X)
        return nn.Softmax(dim = 1)(output)
        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################

        


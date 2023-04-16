import os
import argparse
import torch
import torch.nn as nn
from dataset import SignDigitDataset
from torch.utils.data import DataLoader
from utils import *
from model import MLP
from torch.utils.tensorboard import SummaryWriter
import numpy as np


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

parser = argparse.ArgumentParser()
# Hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100, required=True, help='number of epochs for training')
parser.add_argument('--print_every', type=int, default=10, help='print the loss every n epochs')
parser.add_argument('--img_size', type=int, default=64, help='image input size')
parser.add_argument('--n_classes', type=int, default=6, help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_layers', type=int, required=True, nargs='+',
                    help='number of units per layer (except input and output layer)')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh' , 'sigmoid' , 'none'], help='activation layers')
parser.add_argument('--init_model', type=str , default= 'no')
parser.add_argument('--init_type', type=str , default="zero_constant")
parser.add_argument('--dropout', type=float , default= 0.0)
parser.add_argument('--run_name', type=str , default='first_run')

args = parser.parse_args()

# default `log_dir` is "runs" - we'll be more specific here
log_path = 'runs/sign_digits_experiment_'+str(args.run_name)
layout = {
    "ABCDE": {
        "loss": ["Multiline", ["Train Loss", "Test Loss"]],
        "accuracy": ["Multiline", ["Train ACC", "Test ACC"]],
    },
}
writer = SummaryWriter(log_path)
if args.dropout != 0:
    writer.add_custom_scalars(layout)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

#####################################################################################
# TODO: Complete the script to do the following steps                               #
# 0. Create train/test datasets
# 1. Create train and test data loaders with respect to some hyper-parameters       #
# 2. Get an instance of your MLP model.                                             #
# 3. Define an appropriate loss function (e.g. cross entropy loss)                  #
# 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
# 5. Implement the main loop function with n_epochs iterations which the learning   #
#    and evaluation process occurred there.                                         #
# 6. Save the model weights                                                         #
#####################################################################################


# 0. creating train_dataset and test_dataset
train_dataset = SignDigitDataset(root_dir='data/',
                                 h5_name='train_signs.h5',
                                 train=True,
                                 transform=get_transformations(64))

test_dataset = SignDigitDataset(root_dir='data/',
                                h5_name='test_signs.h5',
                                train=False,
                                transform=get_transformations(64))

# 1. Data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# 2. get an instance of the model
C, H, W = 3, args.img_size, args.img_size
input_size = C*H*W
num_classes = args.n_classes
units = [input_size]
for layer in args.hidden_layers:
    units.append(layer)
units.append(num_classes)

model = MLP(units=units, hidden_layer_activation=args.activation , dropout = args.dropout).to(device)
if args.init_model == 'yes':
    model = init_weights(model, args.init_type)

# 3, 4. loss function and optimizer
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)


# 5. Train the model

for epoch in range(args.n_epochs):
    n_train_batches = 0
    n_test_batches = 0

    total_train = 0
    total_test = 0
    
    train_running_loss, test_running_loss = 0.0, 0.0
    train_running_acc, test_running_acc = 0.0, 0.0

    # Train Part
    for batch in train_loader:
        model.train()
        n_train_batches += 1
        images , labels = batch['image'].to(device) , batch['label'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        images = torch.flatten(images, start_dim = 1) 
        _ , labels = torch.max(labels , 1)
        
        outputs = model(images)
        
        train_loss = loss_function(outputs, labels.long())
        train_running_loss += train_loss         
        _, preds = torch.max(outputs, 1)

        train_running_acc += (preds == labels).float().sum()

        # backward + optimize only if in training phase
        train_loss.backward()
        optimizer.step()
        total_train += images.shape[0]
    
    # Test Part
    for batch in test_loader:
        model.eval()
        n_test_batches += 1

        images , labels = batch['image'].to(device) , batch['label'].to(device)
        images = torch.flatten(images, start_dim = 1)
        _ , labels = torch.max(labels , 1)
        
        outputs = model(images)
        test_loss = loss_function(outputs, labels.long())
        test_running_loss += test_loss         
        _, preds = torch.max(outputs, 1)

        test_running_acc += (preds == labels).float().sum()
        total_test += images.shape[0]
    
    
    # epoch info
    epoch_train_loss = train_running_loss / n_train_batches
    epoch_test_loss = train_running_loss / n_test_batches

    epoch_train_acc = train_running_acc / total_train
    epoch_test_acc = test_running_acc / total_test 
    
    # ...log the running loss
    
    writer.add_scalar('Train Loss',epoch_train_loss, epoch)
    writer.add_scalar('Test Loss', epoch_test_loss , epoch)
    
    writer.add_scalar('Train ACC',epoch_train_acc , epoch)
    writer.add_scalar('Test ACC', epoch_test_acc, epoch)
    
    if epoch % args.print_every == 0:
        # You have to log the accuracy as well
        print('Epoch [{}/{}]:\t Train Loss: {:.4f}, Test Loss: {:.4f}, Train ACC: {:.4f}, Test ACC: {:.4f}'.format(epoch + 1,
                                                                               args.n_epochs,
                                                                               epoch_train_loss,
                                                                               epoch_test_loss , 
                                                                               epoch_train_acc,
                                                                               epoch_test_acc))

#####################################################################################
#                                 END OF YOUR CODE                                  #
#####################################################################################


# save the model weights
checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
    
    https://beheshtiyan.ir/what-is-social-anxiety-and-how-is-it-treated/
    https://beheshtiyan.ir/what-is-social-anxiety-and-how-is-it-treated/
from model import *
from torch_helper import *

import matplotlib.pyplot as plt
import numpy as np


def train(args, x_train, y_train, x_test, y_test, colours, model_mode=None, model=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    #####################################################################################
    # TODO: Implement this function to train model and consider the below items         #
    # 0. read the utils file and use 'process' and 'get_rgb_cat' to get x and y for     #
    #    test and train dataset                                                         #
    # 1. Create train and test data loaders with respect to some hyper-parameters       #
    # 2. Get an instance of your 'model_mode' based on 'model_mode==base' or            #
    #    'model_mode==U-Net'.                                                           #
    # 3. Define an appropriate loss function (cross entropy loss)                       #
    # 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
    # 5. Implement the main loop function with n_epochs iterations which the learning   #
    #    and evaluation process occurred there.                                         #
    # 6. Save the model weights                                                         #
    # Hint: Modify the predicted output form the model, to use loss function in step 3  #
    #####################################################################################
    """
    Train the model
    
    Args:
     model_mode: String
    Returns:
      model: trained model
    """
    torch.set_num_threads(5)
    torch.autograd.set_detect_anomaly(True)
    # Numpy random seed
    np.random.seed(args.seed)

    # Save directory
    save_dir = "outputs/" + args.experiment_name

    print("Transforming data...")
    # Get X(grayscale images) and Y(the nearest Color to each pixel based on given color dictionary)
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input, category_id=args.category_id)
    train_rgb_cat = rgb2label(train_rgb, colours, args.batch_size)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input, category_id=args.category_id)
    test_rgb_cat = rgb2label(test_rgb, colours, args.batch_size)

    # LOAD THE MODEL
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    if model_mode == 'base':
        model = BaseModel(kernel = args.kernel_size, num_filters = args.num_filters, num_colors = args.num_colors, in_channels=1, padding=1)
    elif model_mode == 'U-Net':
        model = CustomUNET(kernel = args.kernel_size , num_filters = args.num_filters, num_colors = args.num_colors, in_channels=1, out_channels=3)
    elif model_mode == 'ResidualU-Net':
        model = ResidualUNET(kernel = args.kernel_size , num_filters = args.num_filters, num_colors = args.num_colors, in_channels=1, out_channels=3)
    
    if args.gpu == True:
        model = model.to(device)

    # LOSS FUNCTION and Optimizer
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = args.learning_rate)
    
    
    # Create the outputs' folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []
    valid_losses = []
    valid_accs = []
    min_val_loss = np.Inf
    iter = 0

    # Training loop
    for epoch in range(args.epochs):
        # Train the Model
        model.train()  # Change model to 'train' mode
        losses = []
        corrects = 0
        all_samples = 0
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            # Convert numpy array to pytorch tensors
            images, labels = get_torch_vars(xs, ys, args.gpu)

            # Forward + Backward + Optimize
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################
            optimizer.zero_grad()
            
            outputs = model(images)
            this_labels = labels.squeeze(1)
            loss = loss_function(outputs , this_labels )
            losses.append(loss.detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            preds = preds.unsqueeze(1)
            #print(preds.shape , labels.shape)
            
            #corrects += sum(preds.view(-1) == labels.view(-1))
            corrects += (preds == labels).sum()

            #print(corrects , corrects.shape)
            
            all_samples += images.shape[0] * images.shape[2] * images.shape[3]
            
            
            
        # Calculate and Print training loss for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################
        epoch_loss = sum(losses)/len(losses)
        train_losses.append(epoch_loss)
        
        
        # Evaluate the model
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################
        train_acc = corrects/all_samples
        print("Epoch:" , epoch)
        print("Train loss: %.4f"%epoch_loss.item() , "Train acc: %.4f"%train_acc )

        
        # Calculate and Print (validation loss, validation accuracy) for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################
        val_loss , val_acc = run_validation_step(model, test_grey , test_rgb_cat , args , loss_function)
        valid_losses.append(val_loss)
        
        print("Val loss: %.4f"%val_loss.item() , "Val acc: %.4f"%val_acc )
        
        ########## Early stopping
        if val_loss < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = val_loss
        else:
            epochs_no_improve += 1
        
        
        plot(xs, ys, preds.detach().cpu().numpy(), colours,
         save_dir + '/train_%d.png' % epoch,
         args.visualize)
        
        print("--------------")
        if epoch > 5 and epochs_no_improve == args.n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue
        
    # Plot training-validation curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")
     
    
     
    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), args.checkpoint)

    return model

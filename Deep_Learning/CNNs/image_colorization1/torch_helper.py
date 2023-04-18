from torch.autograd import Variable
from utils import *


# You can add any function here if needed in the training process

def get_batch(x, y, batch_size):
    """
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    """
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i + batch_size, :, :, :]
        batch_y = y[i:i + batch_size, :, :, :]
        yield batch_x, batch_y


def get_torch_vars(xs, ys, gpu=False):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tensor): greyscale input
      ys (int numpy tensor): categorical labels
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)


# You can implement the model evaluation process in this function
def run_validation_step(model, test_grey , test_rgb_cat , args , loss_function):
    ##############################################################################################
    #                                            YOUR CODE                                       #
    #  return val_loss, val_acc                                                                  #
    ##############################################################################################
    model.eval()
    valid_losses = []
    valid_accs = []
    corrects = 0
    all_samples=0
    for i, (xs, ys) in enumerate(get_batch(test_grey,
                                               test_rgb_cat,
                                               args.batch_size)):

        images, labels = get_torch_vars(xs, ys, args.gpu)
            
        outputs = model(images)
        loss = loss_function(outputs , labels.squeeze(1))
        valid_losses.append(loss.detach().cpu().numpy())
        
        _, preds = torch.max(outputs, 1)
        preds = preds.unsqueeze(1)        
        corrects += (preds == labels).sum()
        
        all_samples += images.shape[0] * images.shape[2] * images.shape[3]
                
    return (sum(valid_losses) / len(valid_losses)) , (corrects/all_samples)
    

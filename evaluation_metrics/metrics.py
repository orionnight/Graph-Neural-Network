import torch

##### Dice ####################

def dice(y_true, y_pred):
    y_true = y_true.view(-1).ge(0.5).type(torch.FloatTensor)
    y_pred = y_pred.view(-1).ge(0.5).type(torch.FloatTensor)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2. * intersection + 1.0) / (union + 1.0)


######### Dice all ###############

def dice_all(y_true, y_pred):
    y_true = y_true.max(0)[1]
    y_pred = y_pred.max(0)[1]
    y_true = y_true.view(-1).gt(0.5).type(torch.FloatTensor)
    y_pred = y_pred.view(-1).gt(0.5).type(torch.FloatTensor)
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2. * intersection + 1.0) / (union + 1.0)

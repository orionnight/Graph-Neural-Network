import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import pdb
from torch.autograd import Variable


def load_data(path):

    data = sio.loadmat(path)
    features = torch.FloatTensor(np.concatenate([data['X'][:,0:3],data['C']],axis=1))
    labels = torch.FloatTensor(data['GT'])


    E1,E2,E3 = sp.find(data['A'])
    X = np.array(features[:,:3])
    Diff = torch.FloatTensor(X[E1,:] - X[E2,:])

    features = features.cuda()
    E1 = torch.FloatTensor(E1).cuda()
    E2 = torch.FloatTensor(E2).cuda()
    Diff = Diff.cuda()
    labels = labels.cuda() 
    features, labels, E1, E2, Diff = Variable(features), Variable(labels), Variable(E1), Variable(E2), Variable(Diff)


    return E1, E2, Diff, features, labels

def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return 100 * correct / len(labels)


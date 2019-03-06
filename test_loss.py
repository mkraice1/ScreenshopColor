from torch import tensor, float
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

loss_fn = MSELoss()

t1 = Variable(tensor([1., 0., 0.]).unsqueeze(0))
t2 = Variable(tensor([0., 22., .9]).unsqueeze(0))

print(t1.shape)

loss = loss_fn( t1, t2 )

print( loss.item() )
from torch import tensor, float
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss, Sigmoid
import numpy as np

mse_loss = MSELoss(reduction="mean")
bce_fn = BCELoss()
sig = Sigmoid()

#t1 = Variable(tensor([[.2, .8, .8], [.2, .8, .8]]))
#t2 = Variable(tensor([[.1, .9, .9], [.1, .9, .9]]))
t1 = Variable(tensor([.8, .8, .0]).unsqueeze(0))
t2 = Variable(tensor([1., 1., .0]).unsqueeze(0))

#t2 = Variable(tensor([0]))

print(sig(t1))

loss = mse_loss( t1, t2 )
print( "BCE then sigmoid: " + str(loss.item()) )
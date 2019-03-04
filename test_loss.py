from torch import tensor
from torch.nn import BCEWithLogitsLoss

loss_fn = BCEWithLogitsLoss

t1 = tensor([.2, .8, .8])
t2 = tensor([.2, .8, .8])

loss = loss_fn( t1, t2 )

print( loss.item() )
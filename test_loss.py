from torch import tensor, float
from torch.nn import BCEWithLogitsLoss
import numpy as np

loss_fn = BCEWithLogitsLoss()

t1 = tensor([-.2, -.8, -.8])
t2 = tensor([22., 22., 2222.])

loss = loss_fn( t1, t2 )

print( loss.item() )
from torch import Tensor
from aihwkit.nn import AnalogLinear

model = AnalogLinear(2, 2)
result = model(Tensor([[0.1, 0.2], [0.3, 0.4]]))
print(result)    


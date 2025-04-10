from torch import Tensor
from aihwkit.nn import AnalogLinear

model = AnalogLinear(2, 2)
result = model(Tensor([[0.1, 0.2], [0.3, 0.4]]))
print(result)    
# from aihwkit.simulator.rpu_base import cuda

# print("AIHWKit compiled with CUDA:", cuda.is_compiled())  # ✅ True
# print("CUDA aware device:", cuda.is_cuda_aware())         # ✅ True or False (depending on setup)


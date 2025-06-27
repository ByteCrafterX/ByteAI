
import torch, platform, sys
print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")


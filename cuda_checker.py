import torch
print("CUDA Version:", torch.version.cuda)
print("PyTorch Built With CUDA:", torch.backends.cuda.is_built())
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))

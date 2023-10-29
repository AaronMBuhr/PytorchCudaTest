import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = total_memory - torch.cuda.memory_allocated(device)
    print(f"Reserved memory on GPU {device}: {reserved_memory / 1024**2:.2f} MiB")
else:
    print("CUDA not available.")

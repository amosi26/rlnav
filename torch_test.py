import torch

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda version (torch build): {torch.version.cuda}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    print(f"gpu device: {idx} - {name}")

    x = torch.rand((1024, 1024), device="cuda")
    y = torch.rand((1024, 1024), device="cuda")
    z = x @ y
    print(f"gpu matmul ok, result shape: {tuple(z.shape)}, device: {z.device}")
else:
    print("No CUDA GPU detected by PyTorch.")

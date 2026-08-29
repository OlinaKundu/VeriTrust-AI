import torch

def get_cuda_device_info() -> dict:
    """
    Returns the active PyTorch compute hardware details.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cuda_version": cuda_version,
        "fp16_enabled": cuda_available,
        "device": "cuda:0" if cuda_available else "cpu"
    }

def print_hardware_summary():
    info = get_cuda_device_info()
    if info["cuda_available"]:
        print(f"[CUDA ACCELERATION ACTIVE]: {info['device_name']} (CUDA {info['cuda_version']}) | Mixed Precision FP16 Enabled")
    else:
        print("[COMPUTE]: Running on CPU.")

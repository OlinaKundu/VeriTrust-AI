HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except Exception as e:
    print(f"[DEVICE WARNING]: PyTorch could not be loaded ({e}). Operating in CPU fallback mode.")
    HAS_TORCH = False

def get_cuda_device_info() -> dict:
    """
    Returns the active PyTorch compute hardware details with fallback safety.
    """
    if not HAS_TORCH:
        return {
            "cuda_available": False,
            "device_name": "CPU (Fallback Mode)",
            "cuda_version": "N/A",
            "fp16_enabled": False,
            "device": "cpu"
        }

    try:
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
    except Exception as e:
        return {
            "cuda_available": False,
            "device_name": f"CPU ({str(e)})",
            "cuda_version": "N/A",
            "fp16_enabled": False,
            "device": "cpu"
        }

def print_hardware_summary():
    info = get_cuda_device_info()
    if info["cuda_available"]:
        print(f"[CUDA ACCELERATION ACTIVE]: {info['device_name']} (CUDA {info['cuda_version']}) | Mixed Precision FP16 Enabled")
    else:
        print(f"[COMPUTE]: Running on {info['device_name']}.")


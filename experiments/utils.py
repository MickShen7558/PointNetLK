import torch
import pynvml

pynvml.nvmlInit()
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
if torch.cuda.device_count() > 1:
    handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
ratio = 1024 ** 2

def print_gpu(s=""):
    if not torch.cuda.is_available():
        print("No cuda available now")
    elif torch.cuda.device_count() > 1:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle1)
        used = (meminfo0.used + meminfo1.used) / ratio
    else:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        used = meminfo0.used / ratio
    print(s+" used: ", used)

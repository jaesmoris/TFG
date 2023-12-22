import torch
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)


def fix_cuda():
    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!

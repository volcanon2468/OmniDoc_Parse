import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python-microservice'))
from app.utils.device_utils import get_device

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available on this machine')
def test_gpu_availability():
    device_str, use_pin_memory = get_device()
    assert device_str == 'cuda'
    assert use_pin_memory is True
    print(f'\nGPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available on this machine')
def test_gpu_availability():
    assert torch.cuda.is_available()
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
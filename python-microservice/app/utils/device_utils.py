import torch
from typing import Tuple

def get_device() -> Tuple[str, bool]:
    try:
        if torch.cuda.is_available():
            return ('cuda', True)
    except Exception:
        pass
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return ('mps', True)
    except Exception:
        pass
    return ('cpu', False)
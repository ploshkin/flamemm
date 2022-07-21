import pickle
from typing import Any

import numpy as np
import scipy.sparse
import torch


class Struct:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype)


def to_np(array: Any, dtype=np.float32) -> np.ndarray:
    if scipy.sparse.issparse(array):
        array = array.todense()
    return np.array(array, dtype=dtype)


def load_flame(path: str) -> Struct:
    with open(path, "rb") as ifile:
        data = pickle.load(ifile, encoding="latin1")
    return Struct(**data)

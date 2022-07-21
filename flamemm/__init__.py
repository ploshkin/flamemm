import functools
import importlib.resources

from .flame import FlameMM as _FlameMM


def _get_resource_path(filename: str) -> str:
    resource_package = importlib.import_module("resources", __package__)
    with importlib.resources.path(resource_package, filename) as path:
        resource_path = str(path)
    return resource_path


FlameMM = functools.partial(
    _FlameMM,
    flame_model_path=_get_resource_path("generic_model.pkl"),
    flame_lmk_embedding_path=_get_resource_path("landmark_embedding.npy"),
    n_shape=100,
    n_exp=50,
)

__all__ = ["FlameMM"]

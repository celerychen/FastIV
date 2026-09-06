"""Import shim for the in-repo ultralytics source.

ultralytics/__init__.py reads the installed torchvision distribution metadata,
which is absent on this machine, so importlib.metadata.version raises
PackageNotFoundError. Fall back to "0.0.0" for any missing distribution.

Import this module before anything that imports ultralytics:

    import torch_env          # noqa: F401
    from ultralytics.nn.tasks import DetectionModel
"""

import importlib.metadata as md

_orig_version = md.version


def _version(name):
    try:
        return _orig_version(name)
    except Exception:
        return "0.0.0"


md.version = _version

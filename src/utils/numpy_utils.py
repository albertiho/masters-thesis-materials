"""Numpy type conversion utilities.

Provides functions to convert numpy types to native Python types for JSON serialization.
This is needed because json.dumps() cannot serialize numpy types
(np.float32, np.bool_, np.int64, etc.).
"""

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization.

    Handles:
    - np.bool_ -> bool
    - np.floating (float16, float32, float64) -> float
    - np.integer (int8, int16, int32, int64) -> int
    - np.ndarray -> list (recursively converted)
    - dict -> dict with values recursively converted
    - list/tuple -> list with items recursively converted

    Args:
        obj: Any object that might contain numpy types.

    Returns:
        Object with all numpy types converted to Python native types.

    Example:
        >>> import numpy as np
        >>> convert_numpy_types({"score": np.float32(0.95), "valid": np.bool_(True)})
        {"score": 0.95, "valid": True}
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

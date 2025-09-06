import numpy as np
import json

import base64


# Custom JSON encoder/decoder for NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy arrays to base64 strings with metadata"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('utf-8'),
                'shape': obj.shape,
                'dtype': obj.dtype.str
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_decode_hook(dct):
    """JSON decode hook to restore NumPy arrays from base64 strings"""
    if isinstance(dct, dict) and dct.get('__numpy_array__'):
        data = base64.b64decode(dct['data'])
        return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
    return dct


def encode_numpy_json(obj):
    """Encode object with NumPy arrays to JSON string"""
    return json.dumps(obj, cls=NumpyEncoder)


def decode_numpy_json(json_str):
    """Decode JSON string and restore NumPy arrays"""
    return json.loads(json_str, object_hook=numpy_decode_hook)

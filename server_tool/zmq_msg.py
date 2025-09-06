import numpy as np
import logging

_logger = logging.getLogger(__name__)


def decode_sync_message(msg, shape_list, dtype_list):
    """Decode sync unit message into list of 6 images"""
    if len(msg) < 8:
        _logger.warning(f"Message too short: {len(msg)} bytes")
        return None, None

    # Extract timestamp (last 8 bytes)
    msg_body, msg_ts = msg[:-8], msg[-8:]
    timestamp = int(np.frombuffer(msg_ts, dtype=np.uint64)[0])

    # Decode images
    images = []
    offset = 0
    for shape, dtype in zip(shape_list, dtype_list):
        # Calculate size in bytes
        size = np.prod(shape) * np.dtype(dtype).itemsize
        if offset + size > len(msg_body):
            _logger.error(f"Message body too short for image data at offset {offset}")
            return None, None

        # Extract and reshape image data
        img_data = np.frombuffer(msg_body[offset:offset + size], dtype=dtype)
        img = img_data.reshape(shape)
        images.append(img)
        offset += size

    return images, timestamp

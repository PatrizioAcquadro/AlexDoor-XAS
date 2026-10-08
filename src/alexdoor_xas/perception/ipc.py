"""Private framed pipes and explicit NumPy array packets for isolated workers."""

import pickle
import struct


def receive(stream):
    header = stream.read(8)
    if not header:
        return None
    size = struct.unpack("!Q", header)[0]
    payload = bytearray()
    while len(payload) < size:
        part = stream.read(size - len(payload))
        if not part:
            raise EOFError("Visual worker pipe closed")
        payload.extend(part)
    return pickle.loads(payload)


def send(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def pack_array(value):
    import numpy as np

    value = np.asarray(value)
    return dict(
        data=np.ascontiguousarray(value).tobytes(), shape=value.shape, dtype=value.dtype.str
    )


def unpack_array(value):
    import numpy as np

    return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()

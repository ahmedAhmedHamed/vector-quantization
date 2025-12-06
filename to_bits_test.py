import numpy as np

import numpy as np

def assignments_to_bytes(assignments, bits):
    arr = np.asarray(assignments, dtype=np.uint32)
    maxv = (1 << bits) - 1
    if np.any(arr > maxv):
        raise ValueError("value exceeds bit-width")

    # Create matrix of bits: shape (N, bits)
    shifts = np.arange(bits-1, -1, -1, dtype=np.uint32)
    bit_matrix = ((arr[:, None] >> shifts) & 1).astype(np.uint8)

    # Flatten bitstream → pack into bytes
    flat_bits = bit_matrix.reshape(-1)
    return np.packbits(flat_bits)


# Example
assignments = [3, 1, 5, 2]
b = assignments_to_bytes(assignments, 3)
print(b)
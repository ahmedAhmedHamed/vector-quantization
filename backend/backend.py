from typing import Tuple, Optional, Union, List, Dict
import numpy as np
from PIL import Image
from backend.VectorQuantizer import VectorQuantizer
import math
import tempfile
import os
import uuid

quantizer = VectorQuantizer()


def save_compressed_data(
        codebook: List[np.ndarray],
        assignments: np.ndarray,  # Already packed bytes
        num_assignments: int,  # Add this parameter
        img_width: int,
        img_height: int,
        block_w: Union[int, float],
        block_h: Union[int, float],
        filepath: str
) -> str:
    # Validate inputs
    if len(codebook) == 0:
        raise ValueError("Codebook is empty")
    if len(assignments) == 0:
        raise ValueError("Assignments list is empty")

    # Convert codebook list to numpy array
    codebook_array = np.array(codebook)

    # assignments is already packed bytes from np.packbits
    assignments_array = assignments

    # Ensure the directory exists and is writable
    file_dir = os.path.dirname(filepath)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    # Convert to absolute path to avoid any path issues
    filepath = os.path.abspath(filepath)

    # Save to .npz file using a file handle to ensure proper writing
    try:
        # Check if we can write to the directory first
        dir_to_check = file_dir if file_dir else os.path.dirname(filepath) or '.'
        if not os.path.exists(dir_to_check):
            os.makedirs(dir_to_check, exist_ok=True)
        if not os.access(dir_to_check, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {dir_to_check}")

        # Remove file if it exists (from previous failed attempts)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

        # Use file handle with np.savez to ensure data is written
        with open(filepath, 'wb') as f:
            np.savez(
                f,
                codebook=codebook_array,
                assignments=assignments_array,
                num_assignments=int(num_assignments),  # Add this
                img_width=int(img_width),
                img_height=int(img_height),
                block_w=int(block_w),
                block_h=int(block_h)
            )
            # Explicitly flush to ensure data is written
            f.flush()
            os.fsync(f.fileno())
    except PermissionError:
        raise
    except OSError as e:
        raise RuntimeError(f"OS error saving to {filepath}: {str(e)}")
    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        raise RuntimeError(f"Failed to save compressed data to {filepath}: {type(e).__name__}: {str(e)}")

    # Verify the file was written and has content
    if not os.path.exists(filepath):
        raise RuntimeError(f"File was not created at {filepath} after np.savez call")

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise RuntimeError(f"File was created but is empty at {filepath}")

    return filepath


def load_compressed_data(filepath: str) -> Dict[str, Union[np.ndarray, int]]:
    data = np.load(filepath)

    # Convert codebook array back to list of numpy arrays
    codebook_array = data['codebook']
    codebook = [codebook_array[i] for i in range(len(codebook_array))]

    # Get assignments as packed bytes
    assignments = data['assignments']

    return {
        'codebook': codebook,
        'assignments': assignments,
        'num_assignments': int(data['num_assignments']),  # Add this
        'img_width': int(data['img_width']),
        'img_height': int(data['img_height']),
        'block_w': int(data['block_w']),
        'block_h': int(data['block_h'])
    }

def reconstruct_image(
    decompressed_vectors: List[np.ndarray],
    img_width: int,
    img_height: int,
    block_w: Union[int, float],
    block_h: Union[int, float]
) -> Image.Image:
    """
    Reconstruct an image from decompressed vectors by reshaping them back into blocks
    and placing them in the correct positions.
    """
    block_width = int(block_w)
    block_height = int(block_h)
    
    # Create empty image array
    reconstructed = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Place blocks back into the image
    block_idx = 0
    for y in range(0, img_height, block_height):
        for x in range(0, img_width, block_width):
            # crop blocks that are too large.
            right = min(x + block_width, img_width)
            bottom = min(y + block_height, img_height)
            
            # Get the decompressed vector for this block
            vector = decompressed_vectors[block_idx]
            
            # Reshape vector back into block
            # Handle edge blocks that might be smaller
            actual_block_w = right - x
            actual_block_h = bottom - y
            # crop and reshape back to square block.
            block = vector[:actual_block_w * actual_block_h].reshape((actual_block_h, actual_block_w))
            
            # Clip values to valid uint8 range and convert
            block = np.clip(block, 0, 255).astype(np.uint8)
            
            # Place block into reconstructed image
            reconstructed[y:bottom, x:right] = block
            
            block_idx += 1
    
    # Convert numpy array back to PIL Image
    return Image.fromarray(reconstructed, mode='L')


def compress_image(
        img: Optional[Image.Image],
        block_w: Union[int, float],
        block_h: Union[int, float],
        amount_of_levels: int
) -> Tuple[Optional[Image.Image], str, float, Optional[str]]:
    if img is None:
        raise ValueError('no img in compress img')

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Compress the image
    codebook, assignments, num_assignments = quantizer.compress(img, block_w, block_h,
                                                                amount_of_levels)  # Unpack num_assignments

    # Decompress to get the reconstructed image
    decompressed_vectors = quantizer.decompress(codebook, assignments, num_assignments)  # Pass num_assignments

    # Reconstruct image from decompressed vectors
    img_width, img_height = img.size
    decompressed_img = reconstruct_image(decompressed_vectors, img_width, img_height, block_w, block_h)

    # Save compressed data to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_filename = 'compressed_' + str(uuid.uuid4()) + '.vq'
    temp_file_path = os.path.join(temp_dir, temp_filename)

    compressed_file_path = save_compressed_data(
        codebook, assignments, num_assignments, img_width, img_height, block_w, block_h, temp_file_path
    )


    # Verify file was created and has content (save_compressed_data already checks this, but double-check)
    if not os.path.exists(compressed_file_path):
        raise RuntimeError(f"Failed to create compressed file at {compressed_file_path}")

    # Check file size to ensure it's not empty
    file_size = os.path.getsize(compressed_file_path)
    if file_size == 0:
        raise RuntimeError(f"Compressed file was created but is empty at {compressed_file_path}")

    # Ensure we return an absolute path for Gradio
    compressed_file_path = os.path.abspath(compressed_file_path)

    # Calculate compression ratio
    # Original size: width * height * 1 byte per pixel (grayscale)
    original_size = img_width * img_height * 1

    # Compressed size:
    # - Codebook: len(codebook) * vector_dim * 8 bytes (float64)
    vector_dim = len(codebook[0]) if len(codebook) > 0 else 0
    codebook_size_bytes = len(codebook) * vector_dim * 8

    # - Assignments: len(assignments) * bits_per_index
    # Each assignment index needs ceil(log2(len(codebook))) bits
    if len(codebook) > 0:
        bits_per_index = math.ceil(math.log2(len(codebook)))
        assignments_size_bits = len(assignments) * bits_per_index
        assignments_size_bytes = assignments_size_bits / 8
    else:
        assignments_size_bytes = 0

    compressed_size = codebook_size_bytes + assignments_size_bytes

    # Compression ratio = original / compressed
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    # Format codebook info as text
    num_blocks = len(assignments)
    codebook_text = f"""Codebook Information:
Codebook Size: {len(codebook)}
Vector Dimension: {vector_dim} ({int(block_w)}x{int(block_h)} blocks)
Number of Blocks: {num_blocks}
Image Dimensions: {img_width}x{img_height}

Compression Statistics:
Original Size: {original_size:,} bytes
Compressed Size: {compressed_size:,.2f} bytes
  - Codebook: {codebook_size_bytes:,.2f} bytes
  - Assignments: {assignments_size_bytes:,.2f} bytes
Compression Ratio: {compression_ratio:.2f}:1

First 3 Codebook Entries (sample):
"""
    # Add first few codebook entries as examples
    for i in range(min(3, len(codebook))):
        codebook_text += f"  Entry {i}: {codebook[i][:min(8, len(codebook[i]))].round(2).tolist()}...\n"

    return decompressed_img, codebook_text, compression_ratio, compressed_file_path


def decompress_image(
        compressed_file_path: Optional[str]
) -> Optional[Image.Image]:
    if compressed_file_path is None:
        return None

    # Load compressed data from file
    data = load_compressed_data(compressed_file_path)
    codebook = data['codebook']
    assignments = data['assignments']
    num_assignments = data['num_assignments']  # Get num_assignments
    img_width = data['img_width']
    img_height = data['img_height']
    block_w = data['block_w']
    block_h = data['block_h']

    # Decompress using quantizer
    decompressed_vectors = quantizer.decompress(codebook, assignments, num_assignments)  # Pass num_assignments

    # Reconstruct image from decompressed vectors
    decompressed_img = reconstruct_image(decompressed_vectors, img_width, img_height, block_w, block_h)

    return decompressed_img

def run_operation(
        operation: str,
        img: Optional[Image.Image] = None,
        compressed_file: Optional[str] = None,
        block_w: Union[int, float] = None,
        block_h: Union[int, float] = None,
        amount_of_levels: Union[int, float] = None
) -> Tuple[Optional[str], Optional[float], Optional[Image.Image], Optional[str]]:
    if operation == "Compression":
        if amount_of_levels is None:
            amount_of_levels = 6  # Default value
        amount_of_levels = int(amount_of_levels)
        decompressed_img, codebook_text, ratio, compressed_file_path = compress_image(img, block_w, block_h, amount_of_levels)
        return (
            codebook_text,
            ratio,
            decompressed_img,
            compressed_file_path
        )
    elif operation == "Decompression":
        decompressed = decompress_image(compressed_file)
        return (
            None,
            None,
            decompressed,
            None
        )
    else:
        return None, None, None, None

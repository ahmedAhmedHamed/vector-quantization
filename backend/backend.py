from typing import Tuple, Optional, Union, List
import numpy as np
from PIL import Image
from backend.VectorQuantizer import VectorQuantizer
import math

quantizer = VectorQuantizer()


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
            right = min(x + block_width, img_width)
            bottom = min(y + block_height, img_height)
            
            # Get the decompressed vector for this block
            vector = decompressed_vectors[block_idx]
            
            # Reshape vector back into block
            # Handle edge blocks that might be smaller
            actual_block_w = right - x
            actual_block_h = bottom - y
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
) -> Tuple[Optional[Image.Image], str, float]:
    if img is None:
        raise ValueError('no img in compress img')

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Compress the image
    codebook, assignments = quantizer.compress(img, block_w, block_h, amount_of_levels)
    
    # Decompress to get the reconstructed image
    decompressed_vectors = quantizer.decompress(codebook, assignments)
    
    # Reconstruct image from decompressed vectors
    img_width, img_height = img.size
    decompressed_img = reconstruct_image(decompressed_vectors, img_width, img_height, block_w, block_h)
    
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
    
    return decompressed_img, codebook_text, compression_ratio


def decompress_image(
        img: Optional[Image.Image],
        codebook
) -> Optional[Image.Image]:
    return img


def run_operation(
        operation: str,
        img: Optional[Image.Image],
        block_w: Union[int, float] = None,
        block_h: Union[int, float] = None,
        amount_of_levels: Union[int, float] = None,
        codebook=None
) -> Tuple[Optional[str], Optional[float], Optional[Image.Image]]:
    if operation == "Compression":
        if amount_of_levels is None:
            amount_of_levels = 6  # Default value
        amount_of_levels = int(amount_of_levels)
        decompressed_img, codebook_text, ratio = compress_image(img, block_w, block_h, amount_of_levels)
        return (
            codebook_text,
            ratio,
            decompressed_img
        )
    elif operation == "Decompression":
        decompressed = decompress_image(img, codebook, "CODEBOOK_PLACEHOLDER")
        return (
            None,
            None,
            decompressed
        )
    else:
        return None, None, None

#!/usr/bin/env python3
"""
Script to print the uncompressed size of a JPEG file.
Uncompressed size = width * height * channels * bytes_per_pixel
"""

import sys
from PIL import Image

def get_uncompressed_size(image_path):
    """Calculate and return the uncompressed size of a JPEG image in bytes."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        channels = len(img.getbands())
        bytes_per_pixel = 1  # JPEG images are typically 8-bit per channel
        
        uncompressed_size = width * height * channels * bytes_per_pixel
        return uncompressed_size
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":

    image_path = r"C:\gitcloned\vector-quantization\original_images\cameraman.jpg"
    size = get_uncompressed_size(image_path)
    print(f"Uncompressed size: {size:,} bytes ({size / (1024*1024):.2f} MB)")


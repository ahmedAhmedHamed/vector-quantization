from typing import Tuple, Optional, Union
from PIL import Image


def compress_image(
    img: Optional[Image.Image], 
    block_w: Union[int, float], 
    block_h: Union[int, float]
) -> Tuple[Optional[Image.Image], str, float]:
    if img is None:
        return None, "CODEBOOK_PLACEHOLDER", 1.0
    compressed_img = img  # no-op
    codebook_text = "CODEBOOK_PLACEHOLDER"
    compression_ratio = 1.0  # placeholder
    return compressed_img, codebook_text, compression_ratio


def decompress_image(
    img: Optional[Image.Image], 
    codebook_text: str
) -> Optional[Image.Image]:
    return img


def run_operation(
    operation: str, 
    img: Optional[Image.Image], 
    block_w: Union[int, float], 
    block_h: Union[int, float]
) -> Tuple[Optional[Image.Image], Optional[str], Optional[float], Optional[Image.Image]]:
    if operation == "Compression":
        compressed_img, codebook, ratio = compress_image(img, block_w, block_h)
        return (
            compressed_img,
            codebook,
            ratio,
            None
        )
    elif operation == "Decompression":
        decompressed = decompress_image(img, "CODEBOOK_PLACEHOLDER")
        return (
            None,
            None,
            None,
            decompressed
        )
    else:
        return None, None, None, None

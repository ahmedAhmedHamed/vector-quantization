def compress_image(img, block_w, block_h):
    # Placeholder: return dummy compressed image & codebook file text
    compressed_img = img  # no-op
    codebook_text = "CODEBOOK_PLACEHOLDER"
    compression_ratio = 1.0  # placeholder
    return compressed_img, codebook_text, compression_ratio


def decompress_image(img, codebook_text):
    # Placeholder: return dummy reconstructed image
    return img  # no-op


def run_operation(operation, img, block_w, block_h):
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

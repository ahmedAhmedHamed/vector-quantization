from PIL import Image
import numpy as np
import os
import pickle
from VectorQuantizer import VectorQuantizer


# Load and convert image to grayscale
img = Image.open("../original_images/cameraman.jpg")
gray_img = img.convert("L")

# Compression parameters
block_w = 4
block_h = 4
amount_of_levels = 6  # This will create a codebook of size 2^6 = 64

print("Starting compression...")
print(f"Original image size: {gray_img.size}")

# Compress the image
vq = VectorQuantizer()
codebook, assignments = vq.compress(gray_img, block_w, block_h, amount_of_levels)

print(f"Compression complete!")
print(f"Codebook size: {len(codebook)}")
print(f"Number of assignments: {len(assignments)}")

# Save compressed artifacts
os.makedirs("../compressed_artifacts", exist_ok=True)

# Save codebook as numpy array
codebook_array = np.array(codebook)
np.save("../compressed_artifacts/codebook.npy", codebook_array)
print("Saved codebook to ../compressed_artifacts/codebook.npy")

# Save assignments as numpy array
assignments_array = np.array(assignments)
np.save("../compressed_artifacts/assignments.npy", assignments_array)
print("Saved assignments to ../compressed_artifacts/assignments.npy")

# Save metadata (image dimensions and block size)
metadata = {
    'img_width': gray_img.size[0],
    'img_height': gray_img.size[1],
    'block_w': block_w,
    'block_h': block_h
}
with open("../compressed_artifacts/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print("Saved metadata to ../compressed_artifacts/metadata.pkl")

# Decompress
print("\nStarting decompression...")
decompressed_vectors = vq.decompress(codebook, assignments)

# Reconstruct image from decompressed vectors
img_width, img_height = gray_img.size
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
decompressed_img = Image.fromarray(reconstructed, mode='L')

# Save decompressed image
os.makedirs("../decompressed_images", exist_ok=True)
decompressed_img.save("../decompressed_images/cameraman_decompressed.jpg")
print("Saved decompressed image to ../decompressed_images/cameraman_decompressed.jpg")

print("\nProcess complete!")

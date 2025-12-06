from typing import Union, Optional, List
import numpy as np

from PIL import Image
import copy


class VectorQuantizer:

    
    def __split_image_into_blocks(self, img: Image.Image, block_w: Union[int, float],
                                  block_h: Union[int, float]) -> List[Image.Image]:
        if img is None:
            return []

        block_width = int(block_w)
        block_height = int(block_h)

        img_width, img_height = img.size

        blocks = []

        for y in range(0, img_height, block_height):
            for x in range(0, img_width, block_width):
                right = min(x + block_width, img_width)
                bottom = min(y + block_height, img_height)
                # 0, 0 is top left
                # rectangle starting from x,y
                # ending at right, bottom.
                block = img.crop((x, y, right, bottom))
                blocks.append(block)

        return blocks

    # convert every image block to an array of numbers
    def __blocks_to_vectors(self, blocks) -> List[np.ndarray]:
        flattened_vectors = []
        for block in blocks:
            block_vector = np.asarray(block)
            flat_vector = block_vector.flatten()
            flattened_vectors.append(flat_vector)
        return flattened_vectors

    def __create_first_level(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Given all flattened image block vectors, compute the first centroid:
        The mean of all vectors.
        """

        # Convert list of vectors → matrix (N vectors × vector_length)
        matrix = np.vstack(vectors)

        #calculate the mean across the the rows (axis=0)(i-th element of every row)
        centroid = np.mean(matrix, axis=0)

        # Return centroid as a NumPy vector
        return centroid


    # splits every vector in the previous level into two new vectors (floor and ceiling).
    def __create_new_level(self, previous_level_blocks: List[np.ndarray]) -> List[np.ndarray]:
        new_level_blocks = []
        for block in previous_level_blocks: # for example: block ->[6.9 7.6 5.2 8.3]
            new_level1 = np.floor(block)    # split to  be [6 7 5 8] and [7 8 6 9]
            new_level2 = np.ceil(block)

            new_level_blocks.append(new_level1)
            new_level_blocks.append(new_level2)
        return new_level_blocks


    # Assign each image vector to the nearest vector in the codebook.
    # Returns: A list of indices corresponding to the assigned code vector for each image_vector.
    def __assign_blocks_to_codebook(
        self, 
        image_vectors: List[np.ndarray], 
        codebook: List[np.ndarray]
    ) -> List[int]:
        # Vectorized implementation for much better performance
        # Convert to numpy arrays for vectorization
        image_matrix = np.array(image_vectors)  # Shape: (num_blocks, vector_dim)
        codebook_matrix = np.array(codebook)    # Shape: (codebook_size, vector_dim)
        
        # Compute all pairwise distances using broadcasting
        # image_matrix: (num_blocks, 1, vector_dim)
        # codebook_matrix: (1, codebook_size, vector_dim)
        # Result: (num_blocks, codebook_size, vector_dim)
        # np.newaxis serves to make the dimensions of both matrices the same so that we can calculate with broadcasting.
        # tl;dr: this creates image-codebook difference.
        differences = np.abs(image_matrix[:, np.newaxis, :] - codebook_matrix[np.newaxis, :, :])
        
        # Sum over vector dimension to get L1 distances: (num_blocks, codebook_size)
        # gets the sum of the distances for a given block.
        #[[1,2], [3,4]] it will return [[3], [7]]
        distances = np.sum(differences, axis=2)
        
        # Find minimum distance index for each block
        assignments = np.argmin(distances, axis=1).tolist()
        
        return assignments  # assignments[i] is the codebook index for image_vectors[i]

        # Function idea: Recalculate Codebook
    
    
    def __recalculate_codebook(
        self, 
        image_vectors: List[np.ndarray], 
        assignments: List[int], 
        old_codebook: List[np.ndarray],
        codebook_size: int
    ) -> List[np.ndarray]:
        # Vectorized implementation for better performance
        image_matrix = np.array(image_vectors)  # Shape: (num_blocks, vector_dim)
        assignments_array = np.array(assignments)  # Shape: (num_blocks,)
        
        new_codebook = []
        
        for i in range(codebook_size):
            # Find all vectors assigned to this centroid
            mask = assignments_array == i
            
            # if no image vector was assigned to this centroid, keep the old 
            if not np.any(mask):
                new_codebook.append(old_codebook[i])
                continue

            # Compute the mean of all vectors in this group using vectorized operations
            assigned_vectors = image_matrix[mask]  # Shape: (num_assigned, vector_dim)
            average_vector = np.mean(assigned_vectors, axis=0)
            
            new_codebook.append(average_vector)

        return new_codebook
   

    def compress(
        self, 
        img: Image.Image, 
        block_w: Union[int, float],
        block_h: Union[int, float], 
        amount_of_levels: int
        ) -> tuple[List[np.ndarray], List[int]]:
        

        source_image_blocks = self.__split_image_into_blocks(img, block_w, block_h)

        image_vectors = self.__blocks_to_vectors(source_image_blocks)

        first_centroid = self.__create_first_level(image_vectors)
        codebook = [np.array(first_centroid)]

        size = 2 ** amount_of_levels
        while len(codebook) < size:
            codebook = self.__create_new_level(codebook)

        previous_codebook = None
        # iterate LBG refinement
        for _ in range(15): # usually needs from 10 to 30 iterations to get the the final codes
            previous_codebook = copy.deepcopy(codebook)
            assignments = self.__assign_blocks_to_codebook(image_vectors, codebook)
            codebook = self.__recalculate_codebook(image_vectors, assignments, codebook, len(codebook))
            if codebook == previous_codebook:
                print('codebook did not change')
                break


        return codebook , assignments

    def decompress(self, codebook: List[np.ndarray], assignments: List[int]):
        ret = []
        for assignment in assignments:
            ret.append(codebook[assignment])
        return ret


if __name__ == '__main__':

    """
        [1, 2], [3, 4],
        [7, 9], [6, 6],
        [4, 9], [10, 10],
        [15, 14], [20, 18],
        [4, 3], [4, 5],
        [17, 16], [18, 18],
        [4, 11], [12, 12],
        [9, 9], [8, 8],
        [1, 4], [5, 6],
    """

    data = np.array([
        [ 1,  2,  7,  9,  4, 11],
        [ 3,  4,  6,  6, 12, 12],
        [ 4,  9, 15, 14,  9,  9],
        [10, 10, 20, 18,  8,  8],
        [ 4,  3, 17, 16,  1,  4],
        [ 4,  5, 18, 18,  5,  6]
    ], dtype=np.uint8)

    img = Image.fromarray(data)

    block_w = 2
    block_h = 2
    amount_of_levels = 2

    codebook, assignments = VectorQuantizer().compress(img, block_w, block_h, amount_of_levels)

    print("--- VQ Compression Results (Lecture Example) ---")
    print(f"Original Image Size: {img.size}")
    print(f"Vector Block Size: {block_w}x{block_h}")
    print(f"Target Codebook Size (2^levels): {2**amount_of_levels}")
    print(f"Final Codebook Size: {len(codebook)}")
    print(f"Total Blocks Assigned: {len(assignments)}")
    
    print("\nFinal Codebook (Centroids):")
    for i, c in enumerate(codebook):
        print(f"  C{i+1}: {c.round(2)}") # Rounding for cleaner output
        
    print("\nFinal Assignments (Index for Blocks 1-9):")
    # Add 1 to indices for human readability if codebook indices are 0-based
    readable_assignments = [int(a) + 1 for a in assignments]
    print(f"  {readable_assignments}")


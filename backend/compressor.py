from typing import Union, Optional, List
import numpy as np

from PIL import Image


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

        assignments = []
        
        # Loop through each individual image vector (Flattened image block)(A)
        for A in image_vectors:

            differences_matrix = []
            
            for C in codebook:  # iterate over every code to get the absolute differnce (distance) between it and the image vector
                difference = A - C
                absolutediff = np.abs(difference)
                sum = np.sum(absolutediff)
                differences_matrix.append(sum)
            
            # find the centroid index with minimum distance
            best_code_index = np.argmin(differences_matrix)

            assignments.append(best_code_index)

            
        return assignments  # assignments[i] is the codebook index for image_vectors[i]

        # Function idea: Recalculate Codebook
    
    
    def __recalculate_codebook(
        self, 
        image_vectors: List[np.ndarray], 
        assignments: List[int], 
        old_codebook: List[np.ndarray],
        codebook_size: int
    ) -> List[np.ndarray]:
        
        # a list of empty lists. one list per code vector
        grouped_vectors = [[] for _ in range(codebook_size)]        

        # Fill each group with the blocks assigned to that centroid
        for vector, centroid_index in zip(image_vectors, assignments):
            grouped_vectors[centroid_index].append(vector)

        new_codebook = []

        # recalculate each centroid

        for i in range(codebook_size):
            
            group = grouped_vectors[i]
            
            # if no image vector was assigned to this centroid, keep the old 
            if len(group) == 0:
                new_codebook.append(old_codebook[i])
                continue

            # compute the mean of all vectors in this group

           

            sum_vector = np.zeros_like(group[0])

            for v in group:
                sum_vector += v

            average_vector = sum_vector / len(group)

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


        # iterate LBG refinement
        for _ in range(15): # usually needs from 10 to 30 iterations to get the the final codes
            assignments = self.__assign_blocks_to_codebook(image_vectors, codebook)
            codebook = self.__recalculate_codebook(image_vectors, assignments, codebook, len(codebook))
        

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


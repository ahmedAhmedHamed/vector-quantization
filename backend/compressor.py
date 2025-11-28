from typing import Union, Optional, List
import numpy as np

from PIL import Image


class VectorQuantizer:

    @staticmethod
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
    def __blocks_to_vectors(self, blocks):
        flattened_vectors = []
        for block in blocks:
            block_vector = np.asarray(block)
            flat_vector = block_vector.flatten()
            flattened_vectors.append(flat_vector)
        return flattened_vectors

    def __create_first_level(self, vectors: List[np.ndarray]) -> List[Union[int, float]]:
        # convert the flatened amtric to a 2-d array
        matrix = np.array(vector)

        #calculate the mean across the the rows (i-th element of every row)
        code_vector_array = np.mean(matrix, axis=0)

        # convert the the NumPy array to a python list of (floats)
        return code_vector_array.tolist()


    # splits every vector in the previous level into two new vectors (floor and ceiling).
    def __create_new_level(self, previous_level_blocks: List[np.ndarray]) -> List[np.ndarray]:
        new_level_blocks = []
        for block in previous_level_blocks: # for example: block ->[6.9 7.6 5.2 8.3]
            new_level1 = np.floor(block)    # split to  be [6 7 5 8] and [7 8 6 9]
            new_level2 = np.ceil(block)

            new_level_blocks.append(new_level1)
            new_level_blocks.append(new_level2)
        return new_level_blocks

    def compress(self, img: Image.Image, block_w: Union[int, float],
                 block_h: Union[int, float], amount_of_levels: int) -> List[Image.Image, dict]:
        ret = None
        source_image_blocks = self.__split_image_into_blocks(img, block_w, block_h)
        current_level = self.__create_first_level(source_image_blocks)
        amount_of_levels -= 1
        while amount_of_levels:
            amount_of_levels -= 1
            current_level = self.__create_new_level(source_image_blocks, current_level)
        return ret

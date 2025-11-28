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
    # made it not flatten.
    def __blocks_to_vectors(self, blocks):
        vectors = []
        for block in blocks:
            block_vector = np.asarray(block)
            vectors.append(block_vector)
        return vectors

    def __create_first_level(self, vectors: List[np.ndarray]) -> List[Union[int, float]]:
        code_vector_array = np.mean(vectors, axis=0)
        return code_vector_array.tolist()


    # splits every vector in the previous level into two new vectors (floor and ceiling).
    def __create_new_level(self, previous_level_blocks: List[np.ndarray], current_level) -> List[np.ndarray]:
        new_level_blocks = []
        for block in previous_level_blocks: # for example: block ->[6.9 7.6 5.2 8.3]
            new_level1 = np.floor(block)    # split to  be [6 7 5 8] and [7 8 6 9]
            new_level2 = np.ceil(block)

            new_level_blocks.append(new_level1)
            new_level_blocks.append(new_level2)
        return new_level_blocks

    def compress(self, img: Image.Image, block_w: Union[int, float],
                 block_h: Union[int, float], amount_of_levels: int) -> List[Image.Image]:
        ret = None
        source_image_blocks = self.__split_image_into_blocks(img, block_w, block_h)
        vectors = self.__blocks_to_vectors(source_image_blocks)
        print(vectors)
        print('-' * 50)
        current_level = self.__create_first_level(source_image_blocks)
        print(current_level)
        amount_of_levels -= 1
        while amount_of_levels:
            amount_of_levels -= 1
            current_level = self.__create_new_level(source_image_blocks, current_level)
        return ret


if __name__ == '__main__':
    data = np.array([
        [1, 2], [3, 4],
        [7, 9], [6, 6],
        [4, 9], [10, 10],
        [15, 14], [20, 18],
        [4, 3], [4, 5],
        [17, 16], [18, 18],
        [4, 11], [12, 12],
        [9, 9], [8, 8],
        [1, 4], [5, 6],
    ], dtype=np.uint8)

    img = Image.fromarray(data)

    block_w = 2
    block_h = 2
    amount_of_levels = 3

    result = VectorQuantizer().compress(img, block_w, block_h, amount_of_levels)

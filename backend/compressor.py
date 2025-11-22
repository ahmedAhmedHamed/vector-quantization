from typing import Union, Optional, List

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

    def __create_first_level(self, source_image_blocks: List[Image.Image]) -> List[Image.Image]:
        pass

    def __create_new_level(self, source_image_blocks: List[Image.Image],
                         previous_level_blocks: List[Image.Image]) -> List[Image.Image]:
        """
        takes the level before it and the source image blocks and creates one more level.
        """
        pass

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

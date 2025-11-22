from typing import Union, Optional, List

from PIL import Image


class VectorQuantizer:

	@staticmethod
	def __split_image_into_blocks(img: Optional[Image.Image], block_w: Union[int, float],
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

from PIL import Image


def crop_to_square(image_path, output_path):

	img = Image.open(image_path)
	width, height = img.size

	new_size = min(width, height)

	left = (width - new_size) // 2
	top = (height - new_size) // 2
	right = left + new_size
	bottom = top + new_size

	img_cropped = img.crop((left, top, right, bottom))

	img_cropped.save(output_path)
	print(f"Cropped image saved to {output_path}")


if __name__ == '__main__':
	crop_to_square("../original_images/cat.webp", "../squared_images/output.webp")

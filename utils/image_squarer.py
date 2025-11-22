from PIL import Image

def crop_to_square(image_path, output_path):
	# Open the image
	img = Image.open(image_path)
	width, height = img.size

	# Determine the size of the square
	new_size = min(width, height)

	# Calculate cropping coordinates to center the square
	left = (width - new_size) / 2
	top = (height - new_size) / 2
	right = left + new_size
	bottom = top + new_size

	# Crop the image
	img_cropped = img.crop((left, top, right, bottom))

	# Save the cropped image
	img_cropped.save(output_path)
	print(f"Cropped image saved to {output_path}")

# Example usage
if __name__ == '__main__':
	crop_to_square("input.jpg", "output.jpg")

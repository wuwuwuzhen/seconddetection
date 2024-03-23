from PIL import Image
import os

def count_openable_images(directory):
    total_images = 0
    openable_images = 0

    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check for .jpg files
            if file.lower().endswith('.jpg'):
                total_images += 1
                try:
                    # Try to open the image
                    with Image.open(os.path.join(root, file)) as img:
                        img.verify()  # Verify that it is a valid image
                        openable_images += 1
                except (IOError, SyntaxError) as e:
                    print(f'Cannot open image: {file}, error: {e}')

    # Print the results
    print(f'Total images: {total_images}')
    print(f'Openable images: {openable_images}')

    # Return the counts as a tuple
    return total_images, openable_images

# Set your directory
image_directory = './picture'  # Update this path to the directory containing your images

# Call the function
count_openable_images(image_directory)

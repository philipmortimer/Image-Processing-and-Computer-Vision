import numpy as np
import cv2

# Conlution function from previous lab
def convolve_image(image, kernel):
    # Checks that function call is valid
    if len(image.shape) != 2:
        raise Exception("Convolution only defined for 2d grayscale images with shape (width, height))")
    if len(kernel.shape) != 2:
        raise Exception("Kernel must be two d")
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise Exception("Kernel width and height must be odd")
    
    # Convoles each pixel
    convolved_image = np.zeros((img_height, img_width))
    kern_pix_right = (kernel_width - 1) // 2
    kern_pix_up = (kernel_height - 1) // 2
    for y in range(0, img_height):
        for x in range(0, img_width):
            val = 0
            for kern_y in range(-kern_pix_up, kern_pix_up + 1):
                for kern_x in range(-kern_pix_right, kern_pix_right + 1):
                    img_x = x - kern_x
                    img_y = y - kern_y
                    # Checks to see that it's not hitting padding part of image. If it is, no need to 
                    # change value
                    if img_y >= 0 and img_y < img_height and img_x >= 0 and img_x < img_width:
                        val += image[img_y, img_x] * kernel[kern_y + kern_pix_up, kern_x + kern_pix_right]
            convolved_image[y, x] = val
    return convolved_image

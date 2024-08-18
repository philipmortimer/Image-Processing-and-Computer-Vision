import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Used to extract colour from image.
# In context of dart boards, the red, green and white components form circles.
# These can be used as part of an input to the hough algorithm to enhance
# detection accuracy by ignoring circles of other colours.
# I ended up not using this approach as it was slow and ineffective for grayscale images.

def get_masked_image(image, show_image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Converts image to HSV space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Defines colour filters
    light_green = (36,0,0)
    dark_green = (86,255,255)
    light_red1 = (0, 70, 50)
    dark_red1 = (10, 255, 255)
    light_red2 = (170, 70, 50)
    dark_red2 = (180, 255, 255)
    light_white = (0, 0, 168)
    dark_white = (172, 111, 255)
    # Shows masked image
    mask_green = cv2.inRange(hsv_img, light_green, dark_green)
    mask_white = cv2.inRange(hsv_img, light_white, dark_white)
    mask_red = cv2.inRange(hsv_img, light_red1, dark_red1) | cv2.inRange(hsv_img, light_red2, dark_red2)
    mask_final = mask_white + mask_green + mask_red
    result = cv2.bitwise_and(image, image, mask=mask_final)
    if show_image:
        plt.subplot(1, 2, 1)
        plt.imshow(mask_final, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.show()
    return [mask_green + mask_red, mask_white]

run = False
if run:
    get_masked_image(cv2.imread("dart.bmp", cv2.IMREAD_COLOR), True)


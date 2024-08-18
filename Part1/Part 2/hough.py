import numpy as np
import cv2
from convolve import convolve_image
import time

# Code used to calculate circular Hough transform for an image

# Helper function for showing an image in opencv
def show_image(title, image, show_images):
    if not show_images:
        return
    cv2.imshow(title, image / np.max(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

# Performs sobel edge detection on image
def sobel(image, show_images):
    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / 8
    dy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]]) / 8
    df_dx = convolve_image(image, dx)
    df_dy = convolve_image(image, dy)
    grad_mag = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Small constant on bottom of fraction to prevent zero division
    grad_dir = np.arctan(df_dy / (df_dx + np.exp(-10)))
    # Shows sobel results
    show_image('Image', image, show_images)
    show_image('df_dx', df_dx, show_images)
    show_image('df_dy', df_dy, show_images)
    show_image('grad_mag', grad_mag, show_images)
    show_image('grad_dir', grad_dir, show_images)
    return (grad_dir, grad_mag, df_dy, df_dx)

# Thresholds an image so all values are 0 or 255 based on threshold
def threshold_image(image, thresh):
    img_height, img_width = image.shape
    ret_image = np.zeros((img_height, img_width))
    for y in range(0, img_height):
        for x in range(0, img_width):
            if image[y, x] > thresh:
                ret_image[y, x] = 255
    return ret_image

# Main hough algorithm. Detects circles in image and returns confidence score
def hough_algo(image, show_images):
    grad_dir, grad_mag, df_dy, df_dx = sobel(image, show_images)
    grad_mag_thresh = threshold_image(grad_mag, 15)
    show_image("Grad mag thresh", grad_mag_thresh, show_images)
    conf = hough_circle_score(grad_mag_thresh, grad_dir, 10, image, show_images)
    if show_images: print("confidence", conf)
    return conf


# Performs hough algorithm to detect circles and return a confidence score [0.0, 1.0].
# Higher confidence => likely dartboard (as circles present)
def hough_circle_score(grad_mag_thresh, grad_dir, thresh_hough, image, show_images):
    # Shrinks image if required to speed up runtime
    scale_factor = 1.0
    grad_mag_thresh = cv2.resize(grad_mag_thresh,(0, 0),fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_AREA)
    grad_dir = cv2.resize(grad_dir,(0, 0),fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_AREA)
    img_height, img_width = grad_mag_thresh.shape
    # Radius length between min_rad and max_rad
    # max_rad is biggest possible radius of circle. Assuming only whole number length radius
    max_rad = round(min(img_width, img_height) * 0.5)
    min_rad = 1
    acc_matrix = np.zeros((img_height, img_width, max_rad))
    for y in range(0, img_height):
        for x in range(0, img_width):
            if grad_mag_thresh[y, x] == 255:
                for rad in range(min_rad, max_rad + 1):
                    # Adds (x_0 = x + rcos(theta), y_0 = y + rsin(theta)) to accumulator
                    y_0 = round(y + rad * np.sin(grad_dir[y, x]))
                    x_0 = round(x + rad * np.cos(grad_dir[y, x]))
                    # if centre is out of bounds, we ignore it in this case.
                    # Note that there are viable alternatives to ignoring it (e.g. extendeding array size etc)
                    if y_0 >= 0 and x_0 >= 0 and y_0 < img_height and x_0 < img_width:
                        acc_matrix[y_0, x_0, rad - min_rad] += 1
                    # Adds (x_0 = x - rcos(theta), y_0 = y - rsin(theta)) to accumulator
                    y_0 = round(y - rad * np.sin(grad_dir[y, x]))
                    x_0 = round(x - rad * np.cos(grad_dir[y, x]))
                    if y_0 >= 0 and x_0 >= 0 and y_0 < img_height and x_0 < img_width:
                        acc_matrix[y_0, x_0, rad - min_rad] += 1
    # Displays hough space. Sums value of radi for each possible x_0, y_0
    show_image('hough space', np.sum(acc_matrix, axis=2), show_images)
    # Thresholds accumulator matrix and loads all circles into a list.
    # Each element is  tuple that describes the circle and how many votes it reveiced.
    circle_vote_list = []
    for rad in range(min_rad, max_rad + 1):
        for y in range(0, img_height):
            for x in range(0, img_width):
                if acc_matrix[y, x, rad - min_rad] <= thresh_hough:
                    acc_matrix[y, x, rad - min_rad] = 0
                else:
                    circle_vote_list.append((y, x, rad - min_rad, acc_matrix[y, x, rad - min_rad]))
    show_image('hough space thresh', np.sum(acc_matrix, axis=2), show_images)
    # Within the context of dartboards, there are 6 (maybe 7) circles that share the same
    # centre. Thus, to measure how likely our image is to be a dartboard, we will assume that
    # the central point is the largest value in the accumulator matrix (an image is
    # unlikely to contain an object where there is a more circular structure.)
    hough_space = np.sum(acc_matrix, axis=2)
    np.unravel_index(np.argmax(hough_space), hough_space.shape)
    [central_point_y, central_point_x] = np.unravel_index(np.argmax(hough_space), hough_space.shape)
    if show_images: print("Circles before ", len(circle_vote_list))
    # Filters all circles that are't sufficiently close to the central point.
    max_dist_to_centre = min(img_width, img_height) / 2
    circle_vote_list = [(y, x, rad_val, votes) 
                      for (y, x, rad_val, votes) in circle_vote_list
                     if np.sqrt(np.square(y - central_point_y) + np.square(x - central_point_x)) <= max_dist_to_centre] 
    if show_images: print("Circles after ", len(circle_vote_list))
    # Sorts circles by radius then number of votes.
    circle_vote_list.sort(key=(lambda x : x[2]))
    circle_vote_list.sort(key=(lambda x : x[3]))
    # Removes circles that have very similar radi as they are probably created by noise and
    # represent one true circle. Circiles with lowest votes in acc matrix that have similar radi
    # are removed first. In case of ties, smaller radi circle are removed first.
    min_radius_difference = 0
    if show_images: print("Circles before ", len(circle_vote_list))
    circle_list_dist_enforced = []
    while(len(circle_vote_list) != 0):
        circle = circle_vote_list.pop(0)
        # Checks to see if a clash exists between current circle and another one
        clash = False
        for i in range(0, len(circle_vote_list)):
            rad_diff_mod = abs((circle_vote_list[i])[2] - circle[2])
            clash = clash or (rad_diff_mod < min_radius_difference)
        # Adds circle to final list if no clash exists
        if clash == False:
            circle_list_dist_enforced.append(circle)
    if show_images: print("Circles after ", len(circle_list_dist_enforced))
    # Scales circle to image dimensions (to undo shrinking effect)
    circle_list_dist_enforced = [(round(y / scale_factor), round(x / scale_factor),
                                   round((rad + min_rad)  / scale_factor)) 
                                 for (y, x, rad, _) in circle_list_dist_enforced]
    # Displays image with circles          
    circle_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for circle in circle_list_dist_enforced:
        # Draws circle
        circle_image = cv2.circle(circle_image, (circle[1], circle[0]), circle[2], (0, 0, 255), 1)
    if show_images:
        cv2.imshow("Circles", circle_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    # Scores overall likelihood of image being a dartboard with a number of simple heuristics
    # Score [0.0, 1.0].
    if (len(circle_list_dist_enforced) == 0):
        return 0.0
    confidence = 0.3
    circle_list_dist_enforced.sort(key=(lambda x : x[2]))
    # Checks to see if there is a large circle taking up most of box
    max_rad = (circle_list_dist_enforced[len(circle_list_dist_enforced) - 1])[2]
    if max_rad >= 0.4 * (min(img_width, img_height) / 2):
        confidence += 0.4
    # Checks to see number of circles. More circles means more confidence
    no_circles = len(circle_list_dist_enforced)
    confidence += no_circles * 0.4
    confidence = max(min(confidence, 1.0), 0.0)
    return confidence


# Used for dev purposes to visualise images generated for Hough space
run = False

if run:
    start_time = time.time()
    image = cv2.imread('Dartboard/dart15.jpg', cv2.IMREAD_GRAYSCALE)
    hough_algo(image, True)
    print("--- %s seconds ---" % (time.time() - start_time))
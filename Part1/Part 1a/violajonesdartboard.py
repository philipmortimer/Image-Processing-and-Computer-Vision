import numpy as np
import cv2
import os
import sys
import argparse

# Viola Jones detection for dartboards
# Code taken from lab4. Used to detect faces and modified for dartboard usecase.

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

# /** Global variables */
cascade_name = "Dartboardcascade/cascade.xml"


def intersection(rect1, rect2):
    # Calculates the intersection between two rectanlges
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    # Calculates relative location of rectangles
    left, right = rect1, rect2
    if x1 > x2:
        left, right = rect2, rect1
    xl, yl, wl, hl = left
    xr, yr, wr, hr = right
    top, bottom = rect1, rect2
    if y1 > y2:
        top, bottom = rect2, rect1
    xt, yt, wt, ht = top
    xb, yb, wb, hb = bottom
    # Calculates x and y overlap
    x_overlap = (xl + wl) - xr
    y_overlap = (yt + ht) - yb
    # If overlaps are negative, cap them at 0
    x_overlap = max(x_overlap, 0)
    y_overlap = max(y_overlap, 0)
    return x_overlap * y_overlap

def rectArea(rect):
    _, _, width, height = rect
    return width * height

def detectAndDisplay(frame):

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    boards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    # 4. Draw box around dartboards found
    thickness = 2
    for i in range(0, len(boards)):
        start_point = (boards[i][0], boards[i][1])
        end_point = (boards[i][0] + boards[i][2], boards[i][1] + boards[i][3])
        colour = (0, 255, 0)
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    # 5. Draw groundtruth boards if applicable
    _, fileName = os.path.split(imageName)
    fileName = fileName.split(".")[0]
    ground_truth = readGroundtruth()
    ground_truth_boards = []
    if fileName in ground_truth:
        for face in ground_truth[fileName]:
            ground_truth_boards.append(face)
            x, y, width, height = face
            frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), thickness)
        # Measures IOU using threshold to detect number of correctly classified images
        thresh = 0.5
        tp = 0
        fp = 0
        boards_classified = [0] * len(ground_truth_boards)
        for face_i in range(0, len(boards)):
            max_i_o_u = 0
            face_index = 0
            for ground_i in range(0, len(ground_truth[fileName])):
                inter = intersection(ground_truth_boards[ground_i], boards[face_i])
                area = rectArea(ground_truth_boards[ground_i]) + rectArea(boards[face_i]) - inter
                i_o_u = inter / area
                if i_o_u > max_i_o_u:
                    max_i_o_u = i_o_u
                    face_index = ground_i
            if max_i_o_u > thresh:
                boards_classified[face_index] = 1
                tp += 1
            else:
                fp += 1
        fn = len(ground_truth_boards) - sum(boards_classified)
        if tp + fn != 0 and tp + fp !=0:
            tpr = tp / (tp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            tpr = 0
        # Calculates f1 score
        if tp != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            # In case where fp is zero. Divide by zero error would occur.
            # However we set f1 score to zero
            f1 = 0
        print("tp", tp, "fp", fp, "fn", fn)
        print("precision", precision, "recall", recall)
        print("TPR", tpr)
        print("F1", f1)
        
                

def readGroundtruth(filename='groundtruth.txt'):
    board_truth = {}
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = float(content_list[1])
            y = float(content_list[2])
            width = float(content_list[3])
            height = float(content_list[4])
            if (img_name in board_truth) == False:
                board_truth[img_name] = []
            board_truth[img_name].append((x, y, width, height))
    return board_truth



# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

imageNames = ["Dartboard/dart" + str(i) + ".jpg" for i in range(0, 16)]
tpr_mean = 0
f1_mean = 0
# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)


# 1. Read Input Image
frame = cv2.imread(imageName, 1)


# 3. Detect Faces and Display Result
detectAndDisplay( frame)

# 4. Save Result Image
cv2.imwrite( "detected.jpg", frame )





import numpy as np
from yolotrainandinfer import load_trained_models, get_boxes
import cv2
import os
import time


# Runs YOLO model on provided images to measure TPR / f1 score.
# Code taken from lab4. Used to detect faces and modified for dartboard usecase.


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

def detectAndDisplay(frame, models_yolo):
	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Runs inferences on yolo models
    boards = get_boxes(cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR), models_yolo)
    # For Yolo Model. Start point is middle of frame, hence shifts box to reflect this
    for i in range(0, len(boards)):
        start_point = (round(boards[i][0] - (boards[i][2] / 2)), round(boards[i][1] - (boards[i][3] / 2)))
        boards[i] = (start_point[0], start_point[1], boards[i][2], boards[i][3])
    # 4. Draw box around faces found
    thickness = 2
    for i in range(0, len(boards)):
        start_point = (boards[i][0], boards[i][1])
        end_point = (round(start_point[0] + boards[i][2]), round(start_point[1] + boards[i][3]))
        colour = (0, 255, 0)
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    # 5. Draw groundtruth boards
    _, fileName = os.path.split(imageName)
    print(fileName)
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
        faces_classified = [0] * len(ground_truth_boards)
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
                faces_classified[face_index] = 1
                tp += 1
            else:
                fp += 1
        fn = len(ground_truth_boards) - sum(faces_classified)
        if tp + fn != 0 and tp + fp !=0:
            tpr = tp / (tp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0
            recall = 0
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
        return (tpr, f1)
        

def readGroundtruth(filename='groundtruth.txt'):
    face_truth = {}
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
            if (img_name in face_truth) == False:
                face_truth[img_name] = []
            face_truth[img_name].append((x, y, width, height))
    return face_truth



# ==== MAIN ==============================================
start_time = time.time()

imageNames = ["Dartboard/dart" + str(i) + ".jpg" for i in range(0, 16)]
tpr_mean = 0
f1_mean = 0

tpr_mean = 0
f1_mean = 0
modles_yolo = load_trained_models()
for idx in range(len(imageNames)):
    imageName = imageNames[idx]
    frame = cv2.imread(imageName, 1)
    tpr, f1 = detectAndDisplay( frame, modles_yolo)
    tpr_mean += tpr
    f1_mean += f1
    cv2.imwrite( "Detected/dart" + str(idx) + "detected.jpg", frame)

tpr_mean /= len(imageNames)
f1_mean /= len(imageNames)

print("TPR mean", tpr_mean)
print("F1 mean", f1_mean)


print("--- %s seconds ---" % (time.time() - start_time))



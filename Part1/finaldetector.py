from yolo_model import load_trained_models, get_boxes
import cv2
import argparse
import os
import sys

# This is the final detector
parser = argparse.ArgumentParser(description='dartboard detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

def detect(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Loads model and performs detection
    model_yolo = load_trained_models()
    dart_boards = get_boxes(cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR), model_yolo)
    # For Yolo Model. Start point is middle of frame, hence shifts box to reflect this
    for i in range(0, len(dart_boards)):
        start_point = (round(dart_boards[i][0] - (dart_boards[i][2] / 2)), round(dart_boards[i][1] - (dart_boards[i][3] / 2)))
        dart_boards[i] = (start_point[0], start_point[1], dart_boards[i][2], dart_boards[i][3])
    thickness = 2
    for i in range(0, len(dart_boards)):
        start_point = (dart_boards[i][0], dart_boards[i][1])
        end_point = (round(start_point[0] + dart_boards[i][2]), round(start_point[1] + dart_boards[i][3]))
        colour = (0, 255, 0)
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    return frame

# Detects image

imageName = args.name

# Checks to see if file exists
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

frame = cv2.imread(imageName, cv2.IMREAD_COLOR)
detected = detect(frame)
cv2.imwrite('detected.jpg', detected)
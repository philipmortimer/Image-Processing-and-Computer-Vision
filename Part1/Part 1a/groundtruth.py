import cv2

# Used to help generate groundtruth values by displaying pixel coordinates for point clicked

dart_folder = "Dartboard"
dart_file_names = ["dart" + str(i)+ ".jpg" for i in range(0, 16)]

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X", x, "Y", y)

def generate_ground_truth():
    for dart_file in dart_file_names:
        img = cv2.imread(dart_folder + "/" + dart_file, cv2.IMREAD_UNCHANGED)
        cv2.imshow('image', img)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_click)
        cv2.waitKey(0)
        print("************************************")
        cv2.destroyAllWindows()

generate_ground_truth()




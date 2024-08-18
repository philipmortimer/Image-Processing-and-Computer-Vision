import cv2
import numpy as np
from ultralytics import YOLO
import os
import random
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil
from ultralytics.utils.torch_utils import strip_optimizer


# Loads negative images
def load_negatives():
    images = []
    for filename in os.listdir('negatives'):
        img = cv2.imread(os.path.join('negatives', filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


# Randomly adds contrast and noise to an image
def contrast_and_noise(img):
    # Adds noise
    if random.random() > 0.85:
        noise_var = 50
        noise = np.random.normal(0, noise_var*random.random(), img.shape)
        img += noise
        np.clip(img, 0., 255.)
    if random.random() > 0.85:
        # Changes contrast and brightness
        alpha = max(0, np.random.normal(1, random.random()))
        beta = min(127, max(-127, np.random.normal(0, 127 / 2)))
        img = cv2.convertScaleAbs(img, alpha, beta)
    return img

def deleteDir(dir):
    # Deletes previous synethetic data
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
           

def generateSynthData(synthFolder):
    # Generates synthetic data of dartboard image
    dart_board_arr = img_to_array(load_img('dart.bmp')) 
    dart_board_arr = dart_board_arr.reshape((1,) + dart_board_arr.shape)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=contrast_and_noise)
    data_items = 4000
    i = 0
    for batch in datagen.flow(dart_board_arr, batch_size=1, 
                              save_to_dir=synthFolder, save_prefix='dartsyn', save_format='jpeg'):
        i += 1
        if i > data_items:
            break

def get_synth_boards(synthFolder):
    # Loads all dart boards (converting them to grascale)
    darts = []
    files = os.listdir(synthFolder)
    for file in files:
        file_path = os.path.join(synthFolder, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        darts.append(img)
    return darts

# Generates images with dart pictures and corresponding labels and writes them to file
def generate_synth_dataset(negatives, darts_synth, data_path):
    # Combines negatives with darts
    no_data_points = 2000
    for i in range(no_data_points):
        background = np.array(random.choice(negatives))
        back_height, back_width = background.shape
        label = []
        # Adds darts to image
        no_darts = max(1, round(abs(np.random.normal(1, 3))))
        rect_list = []
        for j in range(0, no_darts):
            # Scales dart picture
            dart = random.choice(darts_synth).copy()
            height = min(back_height, max(10, abs(round(np.random.normal(back_height * 0.15, back_height * 0.3)))))
            width = min(back_width, max(10, abs(round(np.random.normal(back_width * 0.15, back_width * 0.3)))))
            dart_scale = cv2.resize(dart, (width, height), interpolation = cv2.INTER_AREA)
            # Adds image over foreground
            start_x = random.randint(0, back_width - width)
            start_y = random.randint(0, back_height - height)
            # Adds label
            label.append("0 " + str((start_x + (width / 2)) / back_width) + " " + str((start_y + (height / 2)) / back_height) + " " + str(width / back_width) + " " + str(height / back_height))
            # Simulates occlusion by randomly removing a part of the board for some images
            occluded_area = np.zeros(dart_scale.shape)
            while random.random() <= 0.2:
                occ_width = min(max(1, round(np.random.normal(width / 3, width / 2))), width)
                occ_height = min(max(1, round(np.random.normal(height / 3, height / 2))), height)
                start_occ_x = random.randint(0, width - occ_width)
                start_occ_y = random.randint(0, height - occ_height)
                for y in range(start_occ_y, start_occ_y + occ_height):
                    for x in range(start_occ_x, start_occ_x + occ_width):
                        occluded_area[y, x] = 1
            # Adds image to foreground
            for y in range(start_y, start_y + height):
                for x in range(start_x, start_x + width):
                    # Checks to see if pixel is occluded
                    if occluded_area[y - start_y, x - start_x] == 0:
                        background[y, x] = dart_scale[y - start_y, x - start_x]
            # Adds rectangle for debugging purposes
            x_beg = round((float(label[-1].split(" ")[1]) * back_width) - (width / 2))
            y_beg = round((float(label[-1].split(" ")[2]) * back_height) - (height / 2))
            x_end = round((float(label[-1].split(" ")[3]) * back_width) + x_beg)
            y_end = round((float(label[-1].split(" ")[4]) * back_height) + y_beg)
            rect_list.append([(x_beg, y_beg), (x_end, y_end)])
        
        # Smooths image
        background = cv2.blur(background,(5,5))
        # Writes image to file
        cv2.imwrite(os.path.join(data_path, 'synthDartboards' + str(i) + '.png'), background)
        # Writes label
        lab_file = open(os.path.join(data_path, 'synthDartboards' + str(i) + '.txt'), "w")
        for i in range(len(label)):
            if i != 0: lab_file.write("\n")
            lab_file.write(label[i])
        lab_file.close()

# Splits data in folder to a train, test and validation sub-folder
def split_data_to_val_test_train(data_path):
    val_path = os.path.join(data_path, 'valid')
    test_path = os.path.join(data_path, 'test')
    train_path = os.path.join(data_path, 'train')
    # Deletes contents of these directories if they exist. Creates directories if they dont.
    if os.path.exists(val_path): deleteDir(val_path)
    else: os.mkdir(val_path)
    if os.path.exists(test_path): deleteDir(test_path)
    else: os.mkdir(test_path)
    if os.path.exists(train_path): deleteDir(train_path)
    else: os.mkdir(train_path)
    # Makes image and labels sub directories
    os.mkdir(os.path.join(val_path, 'images'))
    os.mkdir(os.path.join(val_path, 'labels'))
    os.mkdir(os.path.join(test_path, 'images'))
    os.mkdir(os.path.join(test_path, 'labels'))    
    os.mkdir(os.path.join(train_path, 'images'))
    os.mkdir(os.path.join(train_path, 'labels'))
    # Takes each image + label and randomly assigns it a directory
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)
        # Skips directory paths, these shouldn't be dealt with
        if os.path.exists(file_path) and not os.path.isdir(file_path):
            file_name = file.split(".")[0]
            rand = random.random()
            if rand >= 0.0 and rand <= 0.8: new_dir = train_path
            elif rand > 0.8 and rand <= 0.9: new_dir = test_path
            else: new_dir = val_path
            os.rename(os.path.join(data_path, file_name + ".png"), 
                      os.path.join(os.path.join(new_dir, 'images'), file_name + ".png"))
            os.rename(os.path.join(data_path, file_name + ".txt"), 
                      os.path.join(os.path.join(new_dir, 'labels'), file_name + ".txt"))

    

# Uses dart.bmp with negative images to make a dataset to train a classifier on
def generate_training_data():
    synthFolder = 'synthDart'
    data_path = 'synthDataset'
    deleteDir(synthFolder) # Deletes prev synth data
    deleteDir(data_path) # Deletes prev data
    generateSynthData(synthFolder) # Generates synthetic boards
    darts_synth = get_synth_boards(synthFolder)
    negatives = load_negatives()
    generate_synth_dataset(negatives, darts_synth, data_path)
    split_data_to_val_test_train(data_path)


# Trains yolo model
def train_yolo_model():
    model = YOLO("yolov8m.pt")  # loads pretrained model
    model.train(data="data.yaml", epochs=60, imgsz=640, save=True, save_period=20)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    print("Model perfromance")
    print(metrics)
    path = model.export(format="onnx")  # export the model to ONNX format

# Loads trained models to be used for inference
def load_trained_models():
    models = [YOLO("weights/epoch40.pt"), YOLO("weights/otherTrainProcess.pt"), 
              YOLO("weights/epoch20.pt"), YOLO("weights/epoch60.pt")]
    return models


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

# Identifies dartboard regions
def get_boxes(image, models):
    # Loads boxes
    boxes = []
    for model in models:
        # Inference for each model
        results = model(image)
        for result in results:
            box = result.boxes
            print(box.cls)
            if len(box.cls.detach().numpy()) > 0 and box.cls.detach().numpy()[0] == 0:
                box_arr = box.xywh.detach().numpy()[0]
                new_box = [round(box_arr[0]), round(box_arr[1]), 
                            round(box_arr[2]), round(box_arr[3])]
                # Checks to see box is not too similar to already predicted value
                # Uses IOU threshold to check this
                iou_thresh = 0.5
                max_i_o_u = 0
                for box_present in boxes:
                    inter = intersection(new_box, box_present)
                    area = rectArea(new_box) + rectArea(box_present) - inter
                    i_o_u = inter / area
                    max_i_o_u = max(max_i_o_u, i_o_u)
                if max_i_o_u < iou_thresh:
                    boxes.append(new_box)
    # Filters similar boxes using IOU thresholding
    return boxes

# Reduces file size for models saved using checkpoints
def strip_models():
    strip_optimizer('weights/epoch40.pt')
    strip_optimizer('weights/epoch80.pt')
    strip_optimizer('weights/epoch60.pt')
    strip_optimizer('weights/epoch20.pt')

#generate_training_data()
#train_yolo_model()
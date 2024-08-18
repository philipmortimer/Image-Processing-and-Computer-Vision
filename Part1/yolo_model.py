from ultralytics import YOLO
# File used to load trained YOLO models and run inference on them

def load_trained_models():
    models = [YOLO("Part 3/weights/otherTrainProcess.pt"), YOLO("Part 3/weights/epoch40.pt"),
              YOLO("Part 3/weights/epoch20.pt"), YOLO("Part 3/weights/epoch60.pt")]
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


# Gets model predictions for images
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

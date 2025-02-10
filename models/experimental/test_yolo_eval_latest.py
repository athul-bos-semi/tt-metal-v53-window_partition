import numpy as np
from collections import defaultdict
import fiftyone

from pathlib import Path
import shutil
import os
import cv2
import pytest
from models.demos.yolov4.ttnn.yolov4 import TtYOLOv4
from models.demos.yolov4.demo.demo import do_detect

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def iou(pred_box, gt_box):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box[:4]
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box

    # Calculate the intersection area
    ix = max(0, min(x2_pred, x2_gt) - max(x1_pred, x1_gt))
    iy = max(0, min(y2_pred, y2_gt) - max(y1_pred, y1_gt))
    intersection = ix * iy

    # Calculate the union area
    union = (x2_pred - x1_pred) * (y2_pred - y1_pred) + (x2_gt - x1_gt) * (y2_gt - y1_gt) - intersection
    return intersection / union


def calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=3):
    """Calculate mAP for object detection."""
    ap_scores = []

    # Iterate through each class
    for class_id in range(num_classes):
        y_true = []
        y_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = [p for p in pred if p[6] == class_id]
            gt_boxes = [g for g in gt if g[4] == class_id]

            for pred_box in pred_boxes:
                best_iou = 0
                matched_gt = None

                for gt_box in gt_boxes:
                    iou_score = iou(pred_box[:4], gt_box[:4])  # Compare the [x1, y1, x2, y2] part of the box
                    if iou_score > best_iou:
                        best_iou = iou_score
                        matched_gt = gt_box

                # If IoU exceeds threshold, consider it a true positive
                if best_iou >= iou_threshold:
                    y_true.append(1)  # True Positive
                    y_scores.append(pred_box[6])  # Class index as confidence score
                    gt_boxes.remove(matched_gt)  # Remove matched ground truth
                else:
                    y_true.append(0)  # False Positive
                    y_scores.append(pred_box[6])  # Class index as confidence score

            # Ground truth boxes that were not matched are false negatives
            for gt_box in gt_boxes:
                y_true.append(0)  # False Negative
                y_scores.append(0)  # No detection
        if len(y_true) == 0 or len(y_scores) == 0:
            print(f"No predictions or ground truth for class {class_id}")
            continue

        # Calculate precision-recall and average precision for this class
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ap_scores.append(ap)

    # Calculate mAP as the mean of the AP scores
    mAP = np.mean(ap_scores)
    return mAP


def call_ttnn_model(
    ttnn_model=None, imgfile="", n_classes=80, namesfile="models/demos/yolov4/demo/coco.names", device=None
):
    width = 320
    height = 320

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    return do_detect(ttnn_model, sized, 0.3, 0.4, n_classes, device, class_name=namesfile, imgfile=imgfile)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test(device, model_location_generator, reset_seeds):
    dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=50,
    )

    import json

    # Open the JSON file
    with open("/home/ubuntu/fiftyone/coco-2017/info.json", "r") as file:
        # Parse the JSON data
        data = json.load(file)
        classes = data["classes"]

    model_path = model_location_generator("models", model_subdir="Yolo")
    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

    ttnn_model = TtYOLOv4(device, weights_pth)

    ground_truth = []
    predicted_bbox = []
    for i in dataset:
        sample = []
        predicted_temp = call_ttnn_model(ttnn_model=ttnn_model, imgfile=i["filepath"], device=device)
        predicted_bbox.append(predicted_temp)
        for j in i["ground_truth"]["detections"]:
            bb_temp = j["bounding_box"]
            bb_temp[2] += bb_temp[0]
            bb_temp[3] += bb_temp[1]
            bb_temp.append(classes.index(j["label"]))
            sample.append(bb_temp)
        ground_truth.append(sample)

    print("predicted_bbox", predicted_bbox)
    print("length predicted_bbox", len(predicted_bbox))

    class_indices = [box[6] for image in predicted_bbox for box in image]
    num_classes = max(class_indices) + 1

    print(f"Number of classes: {num_classes}")

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    mAPval_50_95 = []
    for iou_threshold in iou_thresholds:
        # Calculate mAP
        mAP = calculate_map(predicted_bbox, ground_truth, num_classes=num_classes, iou_threshold=iou_threshold)
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        mAPval_50_95.append(mAP)

    print("mAPval_50_95", mAPval_50_95)
    mAPval50_95_value = sum(mAPval_50_95) / len(mAPval_50_95)

    print(f"Mean Average Precision for val 50-95 (mAPval 50-95): {mAPval50_95_value:.4f}")

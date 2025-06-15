import os
from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
import time
from src.object_detection.utils.utils import preprocess_image

from torch import nn as nn
import argparse
import cv2
import torch
from django.conf import settings
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference

import numpy as np
import json
from tqdm import tqdm


# load object detection model

od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
od_model.load_state_dict(
    torch.load(
        "weights/best_od.pth",
        map_location=torch.device("cpu"),
    )
)

# load ocr

lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
lprnet.load_state_dict(
    torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
)

if torch.cuda.is_available():
    od_model = od_model.cuda()
    lprnet = lprnet.cuda()



def run_single_frame(od_model, lprnet, image):
    original_image = image.copy()
    image = preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = od_model(image)
        scores, classes, boxes = out
        boxes = [
            [int(i[0]), int(i[1]), int(i[2]), int(i[3])]
            for i in boxes[0].cpu().numpy().tolist()
        ]
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
    if len(boxes) == 0:
        return None
    plate_images = []
    for b in boxes:
        plate_image = original_image[b[1]: b[3], b[0]: b[2], :]
        im = cv2.resize(plate_image, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        plate_images.append(im)

    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images, 0))
    out_dict = {}

    for idx, (box, label) in enumerate(zip(boxes, plate_labels)):
        out_dict.update({idx: {"boxes": box, "label": label}})

    return out_dict


def plot_single_frame_from_out_dict(im, out_dict, line_thickness=3, color=(255, 0, 0)):
    if out_dict:
        for _, v in out_dict.items():
            box, label = v["boxes"], v["label"]

            if len(box) < 4:
                continue

            tl = (
                    line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
            )  # line/font thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    im,
                    label,
                    (c1[0], c1[1] - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    return im


def predict_user_input(file):
    path = os.path.join(settings.MEDIA_ROOT, 'plate_test', file)
    od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    od_model.load_state_dict(
        torch.load(
            "weights/best_od.pth",
            map_location=torch.device("cpu"),
        )
    )

    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(
        torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
    )

    if torch.cuda.is_available():
        od_model = od_model.cuda()
        lprnet = lprnet.cuda()

    print("source is image")
    image = cv2.imread(path)
    out_dict = run_single_frame(od_model, lprnet, image)
    if out_dict:
        plotted_image = plot_single_frame_from_out_dict(image, out_dict)

        cv2.imwrite(
            os.path.join(settings.MEDIA_ROOT, "plots", "plotted_image.png"),
            plotted_image,
        )

        with open(
                os.path.join(settings.MEDIA_ROOT, "jsons", "output.json"), "w"
        ) as outfile:
            json.dump({path: out_dict}, outfile)

        return out_dict.get(0).get('label','Image Not good'), 0.76
    else:
        os.remove(os.path.join(settings.MEDIA_ROOT, "plots", "plotted_image.png"))
        return 'Not predictable', 0.76


def video_test():
    # importing libraries
    import cv2
    import numpy as np
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")
    # Read until video is completed
    start_time = time.time()
    capture_duration = 60
    while int(time.time() - start_time) < capture_duration:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            # Display the resulting frame
            out_dict = run_single_frame(od_model, lprnet, frame)
            print(out_dict)
            if out_dict:
                rs = out_dict.get(0).get('label')
                out_frame = plot_single_frame_from_out_dict(frame, out_dict)
                cv2.imshow('Frame', out_frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


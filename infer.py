import torch
import numpy as np
import cv2
import random

from models import Darknet, load_darknet_weights
from utils.datasets import LoadImages, is_video_file
from utils.utils import scale_coords, load_classes, non_max_suppression, plot_one_box
from utils.parse_config import parse_data_cfg

if __name__ == '__main__':
    import time 

    class Args():
        def __init__(self):
            self.cfg = "cfg/yolov3.cfg"
            self.weights = "weights/yolov3.pt"
            self.data = "data/coco.data"

            self.half = False
            self.device = 'cuda'
            self.nms_thresh = 0.5
            self.score_thresh = 0.3
            self.img_size = 416

            self.source = "data/videos/ice_skate_girl.mp4"

    args = Args()
    device = args.device
    weights = args.weights
    img_size = args.img_size
    source = args.source

    # Initialize model
    model = Darknet(args.cfg, img_size)

    # load model checkpoint
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    print("Loaded %s"%weights)

    model.to(device).eval()

    half = args.half and device != 'cpu'  # half precision only supported on CUDA
    if half: model.half()

    source_is_video = is_video_file(source)
    if source_is_video:
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get classes and colors
    classes = load_classes(parse_data_cfg(args.data)['names'])
    colors = [[random.randint(0, 255) for i in range(3)] for _ in range(len(classes))]

    # Load dataset
    dataset = LoadImages(source, img_size=img_size, half=half)

    cv_wait_time = 1 if source_is_video else 0

    # Run inference
    for path, img, im0, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3: img = img.unsqueeze(0)
        pred = model(img)[0]

        if half: pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, args.score_thresh, args.nms_thresh)

        # Process detections
        s = ""
        for i, det in enumerate(pred):  # detections per image
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        cv2.imshow("img", im0)
        cv2.waitKey(cv_wait_time)

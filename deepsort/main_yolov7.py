# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import sys
import time
import random
import urllib
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

from deep_sort.deepsort import Deepsort_rbc

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    parser.add_argument('--weight', type=str, required=True, help='model name')
    parser.add_argument("--input-video", type=str, required=True, help="path to dataset")
    parser.add_argument('--labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')


    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 2 5 7')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    return parser.parse_args()

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = Ensemble()
        ckpt = torch.load(weightfile, map_location=self.device)
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = self.model.half()
        self.model.eval()

        self.class_names = load_class_names(args.labels)


    def run(self, frame, args):
        sized = cv2.resize(frame, (640, 640))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image)[0]
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=True)

        return pred


class Cosine_Main():
    def __init__(self, args, wt_path='model640.pt'):

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model = torch.load(wt_path)
            self.model.cuda().eval()
        else:
            self.model = torch.load(wt_path, map_location='cpu')
            self.model.eval()


if __name__ == "__main__":
    print(time.time())
    args = get_arguments()

    cap = cv2.VideoCapture(args.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    h, w, c = frame.shape
    print(h, w, c)

    yolov7_main = YOLOv7_Main(args, args.weight)
    outclass = []
    with open(args.labels, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        outclass.append(line)


    cosine_main = Cosine_Main(args)
    dsort = Deepsort_rbc(cosine_main.model, w, h, use_cuda=torch.cuda.is_available())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result.mp4', fourcc, fps, (int(w), int(h)), True)

    print(time.time())
    tracklet = []
    prev_tracklet = []
    c = 0
    while True:
        c += 1
        #print(c)
        ret, frame = cap.read()
        if ret == False:
            print('no video frame')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print('s', time.time())
        result = yolov7_main.run(frame, args)



        '''
        results = yolov7_main.run(frame, args)[0].tolist()
        for result in results:
            l = result[0] * w/640  ## x1
            t = result[1] * h/640  ## y1
            r = result[2] * w/640  ## x2
            b = result[3] * h/640  ## y2

            conf = round(result[4], 2)
            name = outclass[int(result[5])]
            frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
            #frame = cv2.putText(frame, f'{id_num}:{name}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            frame = cv2.putText(frame, f'{name}:{conf}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.jpg', frame)
        '''

        print(time.time())
        tracker = dsort.a_run_deep_sort(frame, result)

#         if c == 10:
#             break

        for track in tracker.tracks:
#                 print('track.is_confirmed(): ', track.is_confirmed())
#                 print('track.time_since_update: ', track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr() #Get the corrected/predicted bounding box
            id_num = str(track.track_id) #Get the ID for the particular track.
            features = track.features #Get the feature vector corresponding to the detection.

            l = bbox[0]  ## x1
            t = bbox[1]  ## y1
            r = bbox[2]  ## x2
            b = bbox[3]  ## y2

            conf = round(track.confidence, 2)
            name = outclass[track.outclass]
            frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
            frame = cv2.putText(frame, f'{id_num}:{name}:{conf}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

    print(time.time())

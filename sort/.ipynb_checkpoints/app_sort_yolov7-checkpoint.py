from __future__ import print_function

import numpy as np
import time
import argparse
import cv2
import datetime
import os
import ffmpeg

import torch
import torch.nn as nn

from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

from sort import *


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
            pred = non_max_suppression(pred, args.det_thr, args.iou_thres, classes=args.classes, agnostic=True)

        return pred



def run(args):
    yolov7_main = YOLOv7_Main(args, args.weight)
    mot_tracker = Sort(max_age=args.max_age,
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold) #create instance of the SORT tracker

    cap = cv2.VideoCapture(args.video)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("sample.mp4", fourcc, fps, (int(width), int(height)), True)

    c = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            print('no video frame')
            break

        c += 1
        print(c)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolov7_main.run(frame, args)

        results = np.asarray(results[0].cpu().detach())
        if results != []:
            for result in results:
                result[0] = result[0] * width/640  ## x1
                result[1] = result[1] * height/640  ## y1
                result[2] = result[2] * width/640  ## x2
                result[3] = result[3] * height/640  ## y2
            results[:, 2:4] += results[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            dets = results
            trackers = mot_tracker.update(dets)

            for track in trackers:
                id_num = track[4] #Get the ID for the particular track.
                l = track[0]  ## x1
                t = track[1]  ## y1
                r = track[2]-track[0]  ## x2
                b = track[3]-track[1]  ## y2

                name = yolov7_main.class_names[int(track[-1])]
                frame = cv2.putText(frame, f'{id_num}:{name}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
            
            sample = frame
        else:
            sample = frame

        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        out.write(sample)
    out.release()



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('-weight', type=str, default='yolov7.pt')
    parser.add_argument("-max-age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=15)
    parser.add_argument("-min-hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("-iou-threshold", help="Minimum IOU for match for Kalman Filter.", type=float, default=0.3)

    # Data
    parser.add_argument('-video', dest='video', action='store', type=str, required=True)
    parser.add_argument('-labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')
    parser.add_argument("-detection-thres", dest='det_thr', type=float, default=0.5)
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-classes', nargs='+', type=int, help='filter by class: -class 0, or -class 0 2 3')

    return parser.parse_args()


if __name__ == '__main__':

    print(time.time())
    args = parse_args()

    run(args)

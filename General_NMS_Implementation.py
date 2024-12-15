import contextlib
import math
import re
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
from matplotlib import pyplot as plt
import cv2

def xywh2xyxy(x):
    """
    > It converts the bounding box from x,y,w,h to x1,y1,x2,y2 where xy1=top-left, xy2=bottom-right

    Args:
      x: the input tensor

    Returns:
      the top left and bottom right coordinates of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


prediction =  torch.rand(1,7,10)
classes = None
agnostic=False
conf_thresh = 0.0
iou_thres = 0.0
max_det = 30000
max_nms = 300
max_wh = 7680

bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
nc = (prediction.shape[1] - 4)  # number of classes
nm = prediction.shape[1] - nc - 4  # number of masks
mi = 4 + nc  # mask start index
xc = prediction[:, 4:mi].amax(1) > conf_thresh  # candidates

prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
for xi, x in enumerate(prediction):  # image index, image inference
    
    x = x[xc[xi]]  # confidence

    if not x.shape[0]:
        continue

    box, cls, mask = x.split((4, nc, nm), 1)

    
    conf, j = cls.max(1, keepdim=True)
    x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thresh]

    # Filter by class
    if classes is not None:
        x = x[(x[:, 5:6] == classes).any(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        continue
    if n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

    print(f"x : {x}\n\n")

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    scores = x[:, 4]  # scores
    boxes = x[:, :4] + c  # boxes (offset by class)
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    i = i[:max_det]  # limit detections


    output[xi] = x[i]

print(f"output : {output}")
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
from torchvision.ops.boxes import box_iou


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



def custom_nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]



prediction =  torch.rand(1,7,8400)
classes = None
agnostic=False
conf_thresh = 0.0
iou_thres = 0.0
max_det = 30000
max_nms = 500
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

    print(f"x size : {x.size()}\n")
    # print(f"x : {x}\n\n") # confidence threshold = 0 & iou_threshold = 0

    # Batched NMS
    # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    c = 0  # classes
    scores = x[:, 4]  # scores
    boxes = x[:, :4] + c  # boxes (offset by class)
    # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    # i = custom_nms(boxes, scores, iou_thres)  # NMS

    i = i[:max_det]  # limit detections


    output[xi] = x[i]

print(f"output size : {output[0].size()}")
# print(f"output : {output}")
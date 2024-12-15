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


# img = cv2.imread("Zentree_Labs\stop.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# stop_data = cv2.CascadeClassifier('Zentree_Labs\stop_data.xml')
# found = stop_data.detectMultiScale(img_gray, minSize =(20, 20))   
# amount_found = len(found)
# if amount_found != 0:
#     for (x, y, width, height) in found:
#         X = x, Y = y, W = width, H = height
#         cv2.rectangle(img_rgb, (x, y), 
#                       (x + height, y + width), 
#                       (0, 255, 0), 5)
        
# plt.subplot(1, 1, 1)
# plt.imshow(img_rgb)
# plt.show()
          

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

# matrx = [[1.], [0., 1., 2.], [1., 1., 4., 4., 7., 8., 9.]]
# prediction =  torch.randn(1,8,7) # tensor of shape (batch_size, num_classes + 4 + num_masks, 8400)

prediction =  torch.rand(1,7,10) # tensor of shape (batch_size, num_classes + 4 + num_masks, 8400)

print(prediction)
print("\n")
# print(prediction[0])

bs = prediction.shape[0]  # batch size
nc = prediction.shape[1] - 4 # number of classes

agnostic=False
conf_thresh = 0.5
iou_thres = .0
max_det = 30000
mi = 4 + nc  # mask start index

# xc = prediction[:, :, 4:mi]
# xc1 = prediction[:, :, 4:mi].amax(1) > conf_thresh
# xc = prediction[:, 4:mi]
xc = prediction[:, 4:mi].amax(1) > conf_thresh

max_wh = 7680  # (pixels) maximum box width and height
max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()



output = [torch.zeros((0, 6), device=prediction.device)] * bs

for xi, x in enumerate(prediction): 

    x = x.transpose(0, -1)[xc[xi]]


    # If none remain process next image
    if not x.shape[0]:
        continue 
    box, cls = x[:, :4], x[:, 4:]


    conf, j = cls.max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thresh]


    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        continue

    x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
    x = x[x[:, 5].argsort(descending = True)]
    print("x : \n")
    print(x)
    print("\n\n")

    curridx = x[0, 5]

    main = []
    fin_ans = []

    for i, xx in enumerate(x):    
        if (xx[5] == curridx):
            main.append(xx)

        else:
            print(f"Current index : {curridx} : \n")

            tens_main = torch.stack(main)             
            c = tens_main[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = tens_main[:, :4] + c, tens_main[:, 4]  # boxes (offset by class), scores
            print(boxes)
            print(c)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            print(f"i vector : {i}\n")

            for a in i:
                fin_ans.append(tens_main[a])

            main.clear()
            curridx = xx[5]
            main.append(xx)
    
    print(f"Current index : {curridx} : \n")

    tens_main = torch.stack(main) 
    c = tens_main[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = tens_main[:, :4] + c, tens_main[:, 4]  # boxes (offset by class), scores
    print(boxes)
    print(c)
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    i = i[:max_det]  # limit detections
    
    print(f"i vector : {i}\n")
    
    for a in i:
        fin_ans.append(tens_main[a])

    # print(fin_ans)
    # print("\n\n")
    fin = torch.stack(fin_ans)
    # print(fin)

    output[xi] = fin

print(output)

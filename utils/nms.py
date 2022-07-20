import torch
import torchvision
import numpy as np


def nms(prediction, iou_thres=0.45, class_agnostic=True):
    # Checks
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height

    output = []
    for x in prediction:  # image index, image inference
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Batched NMS
        if not class_agnostic:
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        else:
            boxes, scores = x[:, :4], x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output.append(x[i].numpy())

    return output
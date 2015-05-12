#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import time


def threshold(image, color, margin=50):
    # Perform macking
    min_ = color - margin
    max_ = color + margin
    min_[ min_ < 0 ] = 0
    max_[ max_ > 255 ] = 255
    mask = cv2.inRange(image, min_, max_)

    # Clean-up with dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.dilate(mask, kernel)

    return mask


def resize(image, target_width):
    # Resize the image to target width
    h, w, c = image.shape
    w_ = int(target_width)
    h_ = int(h * (w_ / w))
    resized = cv2.resize(image, (w_, h_))

    return resized


def graphcut(image, mask):
    # Initialize mask to be graphcut compatible
    mask[mask == 0] = cv2.GC_PR_BGD
    mask[mask == 255] = cv2.GC_PR_FGD

    # Perform graphcut
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # Rescale
    mask[mask == cv2.GC_BGD] = 0
    mask[mask == cv2.GC_PR_BGD] = 0
    mask[mask == cv2.GC_PR_FGD] = 255
    mask[mask == cv2.GC_FGD] = 0

    return mask


def candidates(segment, min_area, min_thesh):
    contour_list, _ = cv2.findContours(segment, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contour_list:
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
        area = w * h
        aspect = w / h
        if area < min_area or aspect >= min_thesh:
            continue
        yield bbox


if __name__ == '__main__':
    AREA_THRESH = 500
    ASPECT_THRESH = 1.0
    TARGET_WIDTH = 500

    time_list = []
    for _ in range(100):
        start_time = time.time()

        image = cv2.imread('3Tg1J4P.jpg')

        color = np.array([30, 0, 150])  # BGR, not RGB
        image = resize(image, TARGET_WIDTH)

        mask = threshold(image, color)

        # The segmentation is expensive.  Without it, you can get into
        # framerate speeds, but performance could suffer
        # mask = graphcut(image, mask)

        for (x, y, w, h) in candidates(mask, AREA_THRESH, ASPECT_THRESH):
            point1 = (x, y)
            point2 = (x + w, y + h)
            cv2.rectangle(image, point1, point2, (0, 255, 0), 2)

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        time_list.append(elapsed_time)

    print("Average elapsed time was %g seconds" % (sum(time_list) / len(time_list), ))

    cv2.imshow('', image)
    cv2.waitKey(0)

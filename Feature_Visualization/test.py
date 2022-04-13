#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 11 14:59:20 2022

@author: Nacriema

Refs:

"""
import cv2


img = cv2.imread("./result_images/layer_26_filter_200.jpg")
# cv2.imshow("Test", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def test_rotate(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w//2, h//2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.5)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Rot", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_rotate(img, 45)

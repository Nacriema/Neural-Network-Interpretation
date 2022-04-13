#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 12 20:15:10 2022

@author: Nacriema

Refs:

"""
import numpy as np
import torch
from torchvision import transforms
import cv2


def preprocess(img):
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return prep(img)


def denormalize(img):
    denorm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                      std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                 transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                      std=[1., 1., 1.]),
                                 ])
    return denorm(img)


def random_transform(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.2)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def create_gauss_2d(size_x, size_y, sigma=1, mu=0.000):
    x, y = np.meshgrid(np.linspace(-2, 2, size_x), np.linspace(-2, 2, size_y))
    dst = np.sqrt(x * x + y * y)
    gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
    gauss_torch = torch.tensor(gauss, dtype=torch.float32, requires_grad=True).cuda()
    return gauss_torch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 11 13:24:16 2022

@author: Nacriema

Refs:

"""
from feature_visual import FilterVisualizer
import os
import torchvision.models as models

os.environ["TORCH_HOME"] = './models/vgg_16_pretrained'
vgg16 = models.vgg16(pretrained=True)

layer = "features.28"
filter = 102

FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2, model=vgg16)
FV.visualize(layer, filter, angle=5, blur=5, lr=0.1, opt_steps=10, use_rotate=True)
FV.plot_image(layer, filter)

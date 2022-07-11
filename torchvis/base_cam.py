#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 07 08:27:11 2022

@author: Nacriema

Refs:

"""
import torch
from functools import reduce
from abc import ABC, abstractmethod


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module):
        self.model = model.eval()

    def _get_module_by_name(self, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, self.model)

    @abstractmethod
    def generate_gradients(self, input_image, target_class):
        pass

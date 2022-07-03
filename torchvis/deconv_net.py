#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jul 03 09:52:12 2022

@author: Nacriema

Refs:

"""
from functools import reduce
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvis.base_cam import BaseCAM
from torchvis.utils import logger


class DeconvNet(BaseCAM):
    def __init__(self, model, layer: str):
        super(DeconvNet, self).__init__(model)
        self.gradients = None

        target_layer = self._get_module_by_name(layer)
        target_layer.register_forward_hook(self.save_gradient)

        # Register forward hook for all ReLU instance in the model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # module.register_forward_hook(self.deconv_op_relu)
                module.register_backward_hook(self.deconv_op_relu_2)

    def save_gradient(self, module, input, output):
        def _store_grad(grad):
            self.gradients = grad.cpu().detach()
        input[0].register_hook(_store_grad)

    # TODO: This is the Guided Backpropagation !!!
    def deconv_op_relu(self, module, input, output):
        def _mod_grad(grad):
            grad = torch.clamp(grad, min=0)
            return grad
        input[0].register_hook(_mod_grad)

    def deconv_op_relu_2(self, module, grad_in, grad_out):
        """
        In my understanding, if we return, then the value we returned will be accepted as the updated grad_in param
        of the module
        """
        return (F.relu(grad_out[0]), )  # This is the Deconv, we clamp the negative gradient w.r.t output of module
        # return (grad_out[0], )  # This is ignore the Backpropagation of the ReLU
        # return (grad_in[0], )  # This is the same as Vanilla Gradient !
        # return (F.relu(grad_in[0]), )  # This is the same as Guided Backpropagation

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        gradient_as_arr = torch.squeeze(self.gradients).data.numpy()
        return gradient_as_arr


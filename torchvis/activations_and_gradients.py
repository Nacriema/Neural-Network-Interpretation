#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 06 17:45:50 2022

@author: Nacriema

Refs:

"""


# https://github.com/jacobgil/pytorch-grad-cam/blob/bb09bd08868abbe106f184e111fc5c7f4d05a0ec/pytorch_grad_cam
# /activations_and_gradients.py
class ActivationsAndGradients:
    """
    Class for extracting activations and registering gradients from targeted intermediate layers.
    Notice about the activation and gradient that the class stored
    For example we put the Layer L
    Then:
        - Activation is the OUTPUT of layer L
        - Gradient is the GRADIENT of the OUTPUT of layer L W.R.T to the 'BASE' gradient

    This is then Use when we are working with Grad-CAM (We need feature maps and gradient w.r.t to them !!!)
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient)
            )

        # self.handles = [target_layer.register_forward_hook(self.save_activation),
        #                 target_layer.register_forward_hook(self.save_gradient)]

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)  # forward pass

    def release(self):
        for handle in self.handles:
            handle.remove()

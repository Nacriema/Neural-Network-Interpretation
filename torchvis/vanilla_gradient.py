#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 07 08:25:53 2022

@author: Nacriema

Refs:

"""
import torch
from torchvis.base_cam import BaseCAM
from torchvis.utils.logger import print_info


class VanillaBackprop(BaseCAM):
    """
    Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model, layer: str):
        super(VanillaBackprop, self).__init__(model)
        self.gradients = None
        self.hook_layers(layer)

    def hook_layers(self, layer):
        def save_gradient(module, input, output):
            def _store_grad(grad):
                self.gradients = grad.cpu().detach()
            input[0].register_hook(_store_grad)

        # Register hook to the first layer (Conv2d)
        first_layer = self._get_module_by_name(layer)
        first_layer.register_forward_hook(save_gradient)

    def generate_gradients(self, input_image, target_class):
        """
        Create gradients
        :param input_image: image contains target_class
        :param target_class: number of target we want to check
        :return:
        """
        print_info("Generating CAM using Vanilla Backpropagation")
        # Forward pass
        model_output = self.model(input_image)
        # Zero grads the model
        self.model.zero_grad()

        # Prepare target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Call backward pass
        model_output.backward(gradient=one_hot_output)

        # Convert Pytorch variable to numpy array, [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jul 21 10:02:13 2022

@author: Nacriema

Refs:

"""
import torch
from torchvis.base_cam import BaseCAM


class GuidedBackprop(BaseCAM):
    def __init__(self, model: torch.nn.Module, layer: str):
        super(GuidedBackprop, self).__init__(model)
        self.gradients = None

        target_layer = self._get_module_by_name(layer)
        target_layer.register_forward_hook(self.save_gradient)

        # Register forward hook for all ReLU instances in model
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(self.guided_op_relu)

    def save_gradient(self, module, input, output):
        def _store_grad(grad):
            self.gradients = grad.cpu().detach()
        input[0].register_hook(_store_grad)

    # This is the Guided Backpropagation
    def guided_op_relu(self, module, input, output):
        def _mod_grad(grad):
            grad = torch.clamp(grad, min=0)
            return grad

        '''
        We can either do input[0].register hook or output.register_hook
        Because: 
        Guided Backprop is the step of clamp the grad w.r.t the input of ReLU (after normal backpropagation through it)
        or clamp w.r.t the output of ReLU (before doing normal backpropagation through it)
        
        But these 2 method have the limitation: We can not skip the normal backpropagation of ReLU !!!
        
        But when we use Module.register_backward_hook, we can do that, se my comment code in DeconvNet
        '''
        input[0].register_hook(_mod_grad)  # SAME: output.register_hook(_mod_grad)

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        gradients_as_arr = torch.squeeze(self.gradients).data.numpy()
        return gradients_as_arr


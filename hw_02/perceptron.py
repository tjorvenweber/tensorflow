#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This class implements a single perceptron.
@author: tjweber
"""

from activation import sigmoid, sigmoidprime

import numpy as np

class Perceptron:
    def __init__(self, input_units):
        # self.bias = np.random.default_rng().uniform(-5,5)
        # self.weights = np.random.default_rng().uniform(-5,5, input_units)
        self.bias = np.random.default_rng().integers(-5,5)
        self.weights = np.random.default_rng().integers(-5,5, input_units)
        self.alpha = 1
        self.drive = 0
        self.activation = 0
        
    def forward_step(self, inputs):
        # catch false number of inputs
        # store inputs for later updating
        self.inputs = inputs
        self.drive = self.bias
        for x, w in list(zip(np.nditer(inputs), np.nditer(self.weights))):
            self.drive += w * x  
        self.activation = sigmoid(self.drive)
        return self.activation

    def update(self, delta):
        gradient = []
        for input in self.inputs:
            gradient.append(input * delta) 
        weights_new = np.subtract(self.weights, self.alpha * np.array(gradient))
        self.weights = weights_new
        self.bias = self.bias - self.alpha * delta
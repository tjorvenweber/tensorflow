#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This class implements a multi-layer perceptron.
@author: tjweber
"""

import numpy as np
from perceptron import Perceptron
from activation import sigmoidprime


class MLP:
    
    def __init__(self, input, hidden, output):

        layers = hidden + [output]
        self.layers = []
        input_units = 0
        for layer in layers:
            input_units = input
            neurons = []

            for i in range(0, layer):
                neurons.append(Perceptron(input_units))

            input = layer
            self.layers.append(neurons)

        
    def forward(self, inputs):
        for layer in self.layers:
            outputs = []
            for perceptron in layer:
                outputs.append(perceptron.forward_step(inputs)) 
            inputs = np.array(outputs)
        self.output = outputs
            
    def backprop(self, labels):

        # compute all deltas
        layers = self.layers[::-1]
        all_deltas = []
        loss = 0
        # loop through layers
        for i in range(0, len(self.layers)):
            if i == 0:
                deltas = []
                # loop through neurons
                for neuron, label in list(zip(layers[i], np.nditer(labels))):
                    loss = label - neuron.activation
                    deltas.append((-loss) * sigmoidprime(neuron.drive))
                all_deltas.append(deltas)
            else:
                deltas = []
                # loop through neurons
                for j in range(0, len(layers[i])):
                    sum = 0
                    # loop through neurons from layer l+1
                    for k in range(0, len(all_deltas[-1])):
                        sum += all_deltas[-1][k] * layers[i-1][k].weights[j]
                    deltas.append(sum * sigmoidprime(layers[i][j].drive))
                all_deltas.append(deltas)

        # update weights
        for deltas, layer in list(zip(all_deltas, layers)):
            for neuron, delta in list(zip(layer, deltas)):
                neuron.update(delta)

        return self.output, loss




            

            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activation functions for neural networks.
@author: tjweber
"""

import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training MLP for gate problem
@author: tjweber
"""

import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# DATA
INPUTS = np.array([[0,0],[0,1],[1,0],[1,1]])
LABELS_AND = np.array([0, 0, 0, 1])
LABELS_OR = np.array([0, 1, 1, 1])
LABELS_NAND = np.array([1, 1, 1, 0])
LABELS_NOR = np.array([1, 0, 0, 0])
LABELS_XOR = np.array([0, 1, 1, 0])

# TRAINING
mlp = MLP(2, [4], 1)

losses = []
accuracies = []
class_counter = 0
correct_counter = 0

for i in range(0, 1000):
    print("epoch ", i)
    epoch_accuracies = []
    epoch_losses = []
    for input, label in (list(zip(INPUTS, np.nditer(LABELS_XOR)))):
        mlp.forward(input)
        output, loss = mlp.backprop(label)

        if (output[0] <= 0.5 and label == 0) or (output[0] > 0.5 and label == 1):
            correct_counter += 1
        class_counter += 1
        accuracy = correct_counter/class_counter
        epoch_accuracies.append(accuracy)
        epoch_losses.append(loss)

        print("loss: ", loss, ";  accuracy: ", accuracy)

    accuracies.append(sum(epoch_accuracies)/len(epoch_accuracies))
    losses.append(sum(epoch_losses)/len(epoch_losses))


# VISUALIZATION
epochs = list(range(0, 1000))
# accuracy
plt.plot(epochs, accuracies)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("XOR Problem: Accuracy over Epochs")
plt.show()

# loss
plt.plot(epochs, losses)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("XOR Problem: Loss over Epochs")
plt.show()
"""
    This class represents a perceptron of one node for input and another for output.
    It calculates the best linear model that classifies the data.
"""

import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    """
    Draws the line calculated by the linear regression function.
    :param x1: the initial point of the line.
    :param x2: the end point of the line.
    """
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.0001)
    ln[0].remove()


def sigmoid(score):
    """
    Calculates a probability based on a score using a sigmoid activation function.
    :param score: score of the point.
    :return: an output between 0 and 1 that defines the probability of a given point using its score.
    """
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    """
    Calculates the loss of given parameters (weight 1, weight 1 and bias) to check if the linear function is the best
    suitable for the data.
    :param line_parameters: the parameters used to build the line (weights and bias).
    :param points: all the points used in the model.
    :param y: the labels of each point in the model.
    :return: the cross entropy (loss) value for the given line parameters.
    """
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(np.log(p).transpose() * y + np.log(1 - p).transpose() * (1 - y)) * (1 / m)
    return cross_entropy


def gradient_descent(line_parameters, points, y, learning_rate):
    """
    The optimization algorithm used to calculate and update the parameters used in the model.
    :param line_parameters: the current line parameters.
    :param points: all the points used in the model.
    :param y: all the labels of each point in the model.
    :param learning_rate: the size of each step realized during the calculation of the new line parameters.
    """
    m = points.shape[0]  # m = number of rows
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        gradient = (points.transpose() * (p - y)) * (learning_rate / m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        # w1x1 + w2x2 + b = 0
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)
        print(calculate_error(line_parameters, points, y))


n_pts = 100  # number of points used in the model.

np.random.seed(0)
bias = np.ones(n_pts)

top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).transpose()
all_points = np.vstack((top_region, bottom_region))

line_parameters = np.matrix([np.zeros(3)]).transpose()

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()

print(calculate_error(line_parameters, all_points, y))

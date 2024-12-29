import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([150, 40])
weights = np.array([[0.2, 0.3],[0.5, 0.7]]).T
true_predictions = np.array([70, 90])
learning_rate = 0.00001

for i in range(50):
    prediction = neural_networks(inp, weights)
    error = get_error(true_predictions, prediction)
    print("Prediction: %s, Weights: %s, Error: %s" % (prediction, weights, error))
    delta = (prediction - true_predictions) * inp * learning_rate
    weights = weights - delta
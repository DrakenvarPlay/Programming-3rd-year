import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([150, 40])
weights = np.array([0.2,0.3])
true_prediction = 1
learning_rate = 0.00007


for i in range(99):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" %(prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta
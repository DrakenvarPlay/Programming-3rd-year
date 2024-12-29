import numpy as np

def neural_networks(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 0.9
weight = 0.9

true_prediction = 0.2
print(get_error(true_prediction, neural_networks(inp, weight)))

for i in range(16):
    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" %(prediction, weight, error))
    delta = (prediction - true_prediction) * inp
    weight = weight - delta
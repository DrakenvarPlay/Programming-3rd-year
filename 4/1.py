import numpy as np

def neural_networks(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 30
weight = 0.9

true_prediction = 90
print(get_error(true_prediction, neural_networks(inp, weight)))

learning_rate = 0.0011
for i in range(8):
    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" %(prediction, weight, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weight = weight - delta
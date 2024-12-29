import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

prediction = neural_networks(np.array([150,40]), [0.2, 0.5])
print(prediction)

true_prediction = 50

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

print(get_error(true_prediction, prediction))

weights = [0.2, 0.3]
prediction = neural_networks(np.array([150,40]), weights)
while get_error(true_prediction, prediction) > 0.001:
    weights[0] += 0.0000001
    prediction = neural_networks(np.array([150, 40]), weights)

print(weights, prediction)
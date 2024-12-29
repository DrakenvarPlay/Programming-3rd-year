import numpy as np

#задаем нейронную сеть с одним входом и несколькими выходами
def neural_networks(inp, weights):
    return inp * weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 400
weights = np.array([0.2,0.3])
true_predictions = np.array([50,120])
learning_rate = 0.00001


for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_predictions, prediction)
    print("Prediction: %s, Weights: %s, Error: %s" %(prediction, weights, error))
    delta = (prediction - true_predictions) * inp * learning_rate
    weights = weights - delta
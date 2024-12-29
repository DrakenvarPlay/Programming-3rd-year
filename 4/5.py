import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))

# def get_error(true_prediction, prediction):
#     return (true_prediction - prediction) ** 2

inp = np.array([
[150, 40],
[170, 80],
[160, 90]
])

learning_rate = 0.00001

true_predictions = np.array([50,120,140])

weights = np.array([0.2,0.3])

for i in range(100):
    error = 0
    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction, prediction)
        print("Prediction: %.10f, True_prediction: %.10f, Weights: %s" %(prediction, true_prediction, weights))
        delta = (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta
    print("Errors: %.10f" % error)
    print("-------------------")
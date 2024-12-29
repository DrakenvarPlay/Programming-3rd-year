import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

# def get_error(true_prediction, prediction):
#     return np.sqrt(np.mean((true_prediction - prediction) ** 2))

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([
[10, 5],
[0, -5],
[2, 6],
])
weights = np.array([0.2,0.3])
true_predictions = np.array([15,-5,8])
learning_rate = 0.01

for i in range(500):
    error = 0
    delta = 0
    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction, prediction)
        print("Prediction: %.10f, True_prediction: %.10f, Weights: %s" %(prediction, true_prediction, weights))
        delta += (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta/len(inp)
    print("Errors: %.10f" % error)
    print("-------------------")

print(neural_networks(np.array([12,4]), weights))
print(neural_networks(np.array([3,-8]), weights))
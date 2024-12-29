import numpy as np


def neuralNetwork(inp, weights):
    prediction_h = inp.dot(weights[0])
    prediction_h2 = prediction_h.dot(weights[1])
    prediction_out = prediction_h2.dot(weights[2])
    return prediction_out

inp = np.array([23, 45])
weight_h_1 = np.random.rand(2)
weight_h_2 = np.random.rand(2)

weight_h2_1 = np.random.rand(2)
weight_h2_2 = np.random.rand(2)

weight_out_1 = np.random.rand(2)
weight_out_2 = np.random.rand(2)


weights_h = np.array([weight_h_1, weight_h_2]).T #транспонируем весовые матрицы
weights_h2 = np.array([weight_h2_1, weight_h2_2]).T
weights_out = np.array([weight_out_1, weight_out_2]).T #транспонируем весовые матрицы

weights = [weights_h, weights_h2, weights_out]


print(neuralNetwork(inp, weights))
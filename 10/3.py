import numpy as np

def relu(x):
    return (x > 0) * x
def relu_deriv(x):
    return x > 0

#реализуем чтото вроде "исключающего или" XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])#входные данные
y = np.array([[0],[1],[1],[0]])#ожидаемый прогноз

#задаем параметры нейронной сети
input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

#фиксируем генератор случайных чисел
np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 100000 #простые вычисления и обучение построим на всем наборе входных данных

for epoch in range(epochs):
    layer_hid = relu(np.dot(x, weight_hid))
# print(layer_hid)
# exit(0)
    layer_out = relu(np.dot(layer_hid, weight_out))
    error = (layer_out - y) ** 2

    layer_out_delta = (layer_out - y) * relu_deriv(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weight_out.T) * relu_deriv(layer_hid)
    #подгоняем веса
    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hidden_delta)
    #каждую 1000 эпоху будем выводить ошибку
    if epoch % 1000 == 0:
        error = np.mean(error)
        print(f"Epoch: {epoch}, Error: {error}")

new_input = np.array([[0,1]])
layer_hid = relu(new_input.dot(weight_hid))
layer_out = relu(layer_hid.dot(weight_out))

print("Предсказание: ", layer_out)
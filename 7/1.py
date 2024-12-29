import numpy as np

np.random.seed(100)

def relu(x):
    return (x > 0) * x

def reluderiv(x):
    return x > 0

inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])
true_prediction = np.array([[10, 20, 15, 20]]).T

layer_hid_size = 4  # Увеличиваем количество скрытых нейронов до 4
layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1

prediction_hid = relu(np.dot(inp[0], weights_hid))
print(prediction_hid)
prediction = prediction_hid.dot(weights_out)
print(prediction)

learning_rate = 0.001  # Изменяем скорость обучения на 0.001
num_epoch = 1000  # Увеличиваем количество эпох до 1000

for i in range(num_epoch):
    layer_out_error = 0  # задаем значение ошибки для вычисления дельты
    for i in range(len(inp)):  # реализуем стохастический градиентный спуск по всем наборам данных
        layer_in = inp[i:i + 1]  # получаем вот такое представление [[15 10]], т.е. не входной вектор значений, а входная матрица, состоящая из одной строки. Это необходимо для дальнейшего вычисления дельты матрицы весовых коэффициентов.
        layer_hid = relu(np.dot(layer_in, weights_hid))
        layer_out = layer_hid.dot(weights_out)
        layer_out_delta = true_prediction[i:i + 1] - layer_out
        layer_hid_delta = layer_out_delta.dot(weights_out.T) * reluderiv(layer_hid)
        layer_out_error += np.sum(layer_out - true_prediction[i:i + 1]) ** 2  # найдем сумму всех ошибок просуммировав значения ошибок всех выходных нейронов. В нашем случае он один, но np.sum делает функцию более универсальной на случай нескольких выходных нейронов

    weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
    weights_hid += learning_rate * layer_in.T.dot(layer_hid_delta)
    print("Predictions: %s, True predictions: %s" % (layer_out, true_prediction[i:i + 1]))
    print("Errors: %.4f" % layer_out_error)
    print("-------------------------------")

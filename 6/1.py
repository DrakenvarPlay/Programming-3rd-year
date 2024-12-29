import numpy as np

def relu(x):
    return (x > 0) * x

# Входные данные
inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])

# Истинные предсказания
true_prediction = np.array([[10, 20, 15, 20]]).T

# Размеры слоев
layer_in_size = len(inp[0])
layer_hid1_size = 5  # Увеличиваем количество нейронов в первом скрытом слое
layer_hid2_size = 4  # Добавляем второй скрытый слой с 4 нейронами
layer_out_size = len(true_prediction[0])

# Инициализация весов
weights_hid1 = 2 * np.random.random((layer_in_size, layer_hid1_size)) - 1
weights_hid2 = 2 * np.random.random((layer_hid1_size, layer_hid2_size)) - 1
weights_out = 2 * np.random.random((layer_hid2_size, layer_out_size)) - 1

# Прямой проход для первого примера
prediction_hid1 = relu(np.dot(inp[0], weights_hid1))
prediction_hid2 = relu(np.dot(prediction_hid1, weights_hid2))
prediction = prediction_hid2.dot(weights_out)

print("Первый скрытый слой:", prediction_hid1)
print("Второй скрытый слой:", prediction_hid2)
print("Предсказание:", prediction)

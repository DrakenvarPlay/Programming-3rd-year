import numpy as np

# Базовый класс для слоя
class Layer(object):
    def __init__(self):
        self.parameters = []  # Параметры класса (наследников класса)

    def get_parameters(self):  # Геттер
        return self.parameters

# Линейный слой
class Linear(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()  # Вызываем базовый конструктор
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0 / input_count)
        self.weight = weight
        self.bias = np.zeros((1, output_count))  # Смещение
        self.parameters.append(self.weight)  # Добавляем вес в параметры
        self.parameters.append(self.bias)  # Добавляем смещение в параметры

    def forward(self, inp):
        return Tensor(inp.data.dot(self.weight) + self.bias)  # Прямое распространение

# Класс для вычисления RMSE
class RMSE:
    def __init__(self):
        pass

    def __call__(self, true_values, predictions):
        return np.sqrt(np.mean((true_values - predictions) ** 2))

# Класс для вычисления MSE (для сравнения)
class MSE:
    def __call__(self, true_values, predictions):
        return np.mean((true_values - predictions) ** 2)

# Класс для тензоров (для упрощения работы с данными)
class Tensor:
    def __init__(self, data, autograd=False):
        self.data = data
        self.autograd = autograd

    def dot(self, other):
        return Tensor(np.dot(self.data, other.data))

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return Tensor(self.data + other)
        elif isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Tensor' and '{}'".format(type(other)))

    def __neg__(self):
        return Tensor(-self.data)  # Поддержка унарного минуса

# Нейросеть
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

# Создание нейросети для перемножения трех чисел
nn_multiply = NeuralNetwork()
nn_multiply.add_layer(Linear(3, 5))  # Входной слой с 3 входами и 5 выходами
nn_multiply.add_layer(Linear(5, 1))   # Выходной слой с 1 выходом

# Пример входных данных
input_data = Tensor(np.array([[1, 2, 3]]), autograd=True)  # Пример входа
output = nn_multiply.forward(input_data)
print("Output for multiplication:", output.data)

# Замена MSE на RMSE
true_values = np.array([[6]])  # Ожидаемое значение
predictions = output.data  # Предсказанное значение

mse = MSE()
rmse = RMSE()

print("MSE:", mse(true_values, predictions))
print("RMSE:", rmse(true_values, predictions))

# Создание нейросети для определения пола по весу и возрасту
nn_gender = NeuralNetwork()
nn_gender.add_layer(Linear(2, 3))  # Входной слой с 2 входами (возраст и вес) и 3 выходами
nn_gender.add_layer(Linear(3, 2))   # Скрытый слой с 3 нейронами и 2 выходами (мужской и женский пол)

# Пример входных данных
input_gender = Tensor(np.array([[25, 70], [30, 80]]), autograd=True)  # Пример входа (возраст и вес)
output_gender = nn_gender.forward(input_gender)
print("Gender Output:", output_gender.data)

# Функция активации (Сигмоида)
class Sigmoid:
    def __call__(self, x):
        return Tensor(1 / (1 + np.exp(-x.data)))  # Используем .data для получения массива

# Линейный слой с функцией активации
class LinearWithActivation(Linear):
    def __init__(self, input_count, output_count, activation):
        super().__init__(input_count, output_count)
        self.activation = activation

    def forward(self, inp):
        return self.activation(super().forward(inp))

# Нейросеть для определения пола с функцией активации
nn_gender_with_activation = NeuralNetwork()
nn_gender_with_activation.add_layer(LinearWithActivation(2, 3, Sigmoid()))  # Входной слой с 2 входами и 3 выходами
nn_gender_with_activation.add_layer(LinearWithActivation(3, 2, Sigmoid()))  # Скрытый слой с 3 нейронами и 2 выходами

output_gender_with_activation = nn_gender_with_activation.forward(input_gender)
print("Gender Output with Sigmoid:", output_gender_with_activation.data)

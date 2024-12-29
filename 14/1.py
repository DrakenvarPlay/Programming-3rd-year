import numpy as np

class Tensor(object):
    _id_counter = 0  # Статическая переменная для генерации ID

    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        if id is None:
            id = Tensor._id_counter
            Tensor._id_counter += 1
        self.id = id
        self.creators = creators if creators is not None else []
        if self.creators:
            for creator in self.creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data, [self], "neg", True)
        return Tensor(-self.data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.T, [self], "T", True)
        return Tensor(self.data.T)

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), [self], "sigmoid", True)
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh", True)
        return Tensor(np.tanh(self.data))

    def relu(self):
        if self.autograd:
            return Tensor(np.maximum(0, self.data), [self], "relu", True)
        return Tensor(np.maximum(0, self.data))

    def __str__(self):
        return str(self.data.__str__())

    def __repr__(self):
        return str(self.data.__repr__())

    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
        if grad_origin is not None:
            if (self.children[grad_origin.id]) > 0:
                self.children[grad_origin.id] -= 1
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        if self.creators and (self.check_grads_from_children() or grad_origin is None):
            if self.operation_on_creation == "+":
                self.creators[0].backward(grad, self)
                self.creators[1].backward(grad, self)
            elif self.operation_on_creation == "-":
                self.creators[0].backward(grad, self)
                self.creators[1].backward(-grad, self)
            elif self.operation_on_creation == "*":
                new = grad * self.creators[1].data
                self.creators[1].backward(grad * self.creators[0].data, self)
                self.creators[0].backward(new, self)
            elif self.operation_on_creation == "dot":
                temp = self.grad.transpose().dot(self.creators[0]).transpose()
                self.creators[1].backward(temp, self)
            elif self.operation_on_creation == "neg":
                self.creators[0].backward(-grad, self)
            elif self.operation_on_creation == "T":
                self.creators[0].backward(grad.transpose(), self)
            elif self.operation_on_creation == "sigmoid":
                sigmoid_grad = self.data * (1 - self.data)
                self.creators[0].backward(grad * sigmoid_grad, self)
            elif self.operation_on_creation == "tanh":
                tanh_grad = 1 - self.data ** 2
                self.creators[0].backward(grad * tanh_grad, self)
            elif self.operation_on_creation == "relu":
                relu_grad = self.data > 0
                self.creators[0].backward(grad * relu_grad, self)


class SGD(object):
    def __init__(self, weights, learning_rate=0.001):  # Уменьшите learning_rate
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        for weight in self.weights:
            if weight.grad is not None:
                print(f"Weight shape: {weight.data.shape}, Grad shape: {weight.grad.data.shape}")  # Отладочный вывод

                # Приведение градиентов к нужной форме
                if weight.data.shape == (3, 1) and weight.grad.data.shape == (3,):
                    weight.grad.data = np.reshape(weight.grad.data, (3, 1))
                elif weight.data.shape == (1,) and weight.grad.data.shape == (3, 1):
                    weight.grad.data = np.mean(weight.grad.data, axis=0)  # Уменьшаем до (1,)

                # Обновляем веса
                weight.data -= self.learning_rate * weight.grad.data

                # Обнуляем градиент для следующей итерации
                weight.grad = Tensor(np.zeros_like(weight.data), autograd=True)

# Обратите внимание на нормализацию данных перед обучением

# Пример использования ReLU
a = Tensor([-1, 2, -3, 4], autograd=True)
b = a.relu()
print(b)  # Ожидаемый вывод: [0, 2, 0, 4]

# Инициализация весов
w1 = Tensor(np.random.randn(3, 1), autograd=True)
b1 = Tensor(np.random.randn(1), autograd=True)

# Обучающие данные
X = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], autograd=True)
y = Tensor([[6], [120], [504]], autograd=True)

# Обучение нейросети
optimizer = SGD([w1, b1], learning_rate=0.01)

for epoch in range(1000):
    # Прямой проход
    y_pred = X.dot(w1) + b1

    # Функция потерь (среднеквадратичная ошибка)
    loss = ((y_pred - y) * (y_pred - y))

    # Обратное распространение
    loss.backward()

    # Обновление весов
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Проверка на тестовых данных
test_input = Tensor([[3, 5, 4]], autograd=True)
test_output = test_input.dot(w1) + b1
print(f"Predicted output: {test_output.data}")
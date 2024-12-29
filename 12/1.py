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

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data, [self], "neg", True)
        return Tensor(-self.data)

    def __str__(self):
        return str(self.data.__str__())

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
            elif self.operation_on_creation == "neg":
                self.creators[0].backward(-grad, self)

# Пример для проверки правильности кода с автоградиентом
a_1 = Tensor([1, 2, 3], autograd=True)
a_2 = Tensor([4, 5, 6], autograd=True)
a_3 = Tensor([7, 8, 9], autograd=True)

a_add_1 = a_1 + a_2
a_sub_1 = a_2 - a_3
a_mul_1 = a_1 * a_3
a_neg_1 = -a_2

a_final = a_add_1 + a_sub_1 + a_mul_1 + a_neg_1

# Проверка градиентов
a_final.backward(Tensor([1, 1, 1]))

print("a_1.grad:", a_1.grad)
print("a_2.grad:", a_2.grad)
print("a_3.grad:", a_3.grad)

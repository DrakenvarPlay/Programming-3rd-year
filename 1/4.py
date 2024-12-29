def neuralNetwork(inp, weights):

    # weights - является вектором векторов, т.е. мы получаем двумерный массив

    # значений весовых коэффициентов или матрицу

    prediction = [0] * len(weights) #Упростим запись количества прогнозов

    for i in range(len(weights)):

        ws=0 #средневзвешенное значение i выходного нейрона

        for j in range(len(inp)):

            ws += inp[j] * weights[i][j]

        prediction[i] = ws

    return prediction



# Посмотрим какие у нас будут входные данные

inp = [50, 165, 45]



weights_1 = [0.2, 0.1, 0.65]

weights_2 = [0.3, 0.1, 0.7] # weights_2 = weights_1 сделает значения на соответствующих выходных нейронах одинаковыми

weights_3 = [0.5, 0.4, 0.34]

weights_4 = [0.4, 0.2, 0.1]



weights = [weights_1, weights_2, weights_3, weights_4]

out = neuralNetwork(inp, weights)
f = (out[0] - out[1]) / abs(out[0] - out[1])
search = 0.01
while abs(out[0] - out[1]) >0.00000000000001:
    weights[1][0] += search * f
    weights[1][1] += search * f
    weights[1][2] += search * f
    out = neuralNetwork(inp, weights)
    pf = f
    f = (out[0] - out[1]) / abs(out[0] - out[1])
    if pf != f: search /= 10


# Добавление еще одного массива weights создает еще один выход


print(out)
print(weights)
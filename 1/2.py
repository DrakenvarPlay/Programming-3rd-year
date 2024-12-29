def neuralNetwork(inps, weights):

    prediction = 0
    list = []
    # Создаем список в котором будут храниться промежуточные значения

    # В цикле перебора массива весов вычисляем средневзвешенное значение

    # посредством скалярного произведения двух векторов

    for i in range(len(weights)):
        list.append(inps[i]*weights[i])
        #добавляем в список каждое промежуточное значение
        prediction += inps[i]*weights[i]

    return prediction, list

# Используем нейросеть
# Создадим два предсказания

out_1 = neuralNetwork([150, 40], [0.3, 0.4])

out_2 = neuralNetwork([80, 60], [0.2, 0.4])

# Выводим результат

print(out_1)

print(out_2)

# запускаем Shift+F10 и проверяем
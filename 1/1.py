# Создаем функцию
def neuralNetwork(inp, weight, bias):
# рассчитываем прогноз
     prediction = inp * weight + bias
     return prediction


print(neuralNetwork(150, 0.4, 3))

#К результату добавится значение bias



# # Используем нейросеть
#
# # Создадим два предсказания
# inputs = [150, 160, 170, 180, 190]
#
# for i in inputs:
#      print(neuralNetwork(i, 0.4))
#
# out_1 = neuralNetwork(160, 0.1)
# out_2 = neuralNetwork(120, 0.7)
# # Входные данные и веса прямопропорционально влияют на выходные данные
# # Выводим результат
# print(out_1)
# print(out_2)
# # запускаем Shift+F10 и проверяем
import numpy as np
arr1 = np.array([1,2,3,4,5])
print(arr1*2)

arr1 = np.random.rand(3,3)
arr2 = np.random.rand(3,3)
print(arr1 * arr2)

arr1 = np.random.randint(0, 100, 10)
print(arr1)
print(arr1[arr1 % 2 == 0])
print(np.mean(arr1))
print(np.std(arr1))
print(np.max(arr1))
print(np.min(arr1))

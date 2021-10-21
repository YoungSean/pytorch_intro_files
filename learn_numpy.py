import numpy as np
#
# arr = np.array([[1,2,3],
#                [3,4,5]], dtype=np.float)
#
# print(arr)
# print("Number of dimension: ",arr.ndim)
# print("Shape: ", arr.shape, " done")
# print("Data type is: ", arr.dtype)

# azeros = np.zeros((2,5))
# print(azeros)
# print("Zeros data type", azeros.dtype)
#
# aone = np.ones((3,4))
# print(aone)
# print("Ones data type", aone.dtype)
#
# print(np.arange(1,10,2))
#
# b = np.arange(12).reshape(3,4)
# print("B data type: ", b.dtype)
# a = np.array([np.pi, 2*np.pi, 20, 40])
# b = np.arange(4)
# #d = [1,2,3]
# c = b<3
# print(c)
#
# a = np.array([[1,1],
#      [0,1]])
# b = np.arange(4).reshape(2,2)
# print(a,"\n",b)
# c = a*b
# print("a*b:\n",c)
# c_dot = a.dot(b)
# print("a mm b:\n", c_dot)

#af = np.random.random(size=(3,4))
# af = np.arange(2,14).reshape(3,4)
# print("numbers: ", af)
# # print("Minimum in the whole array: " ,(np.min(af, axis=1)))
# # print("Maximum in the whole array: ", np.max(af))
# # print("sum of elements: ", np.sum(af, axis=0))
# print("the index of minimum element: ", np.argmin(af, axis=0))
# print("The mean is: ", np.mean(af, axis= 1))
#
# print("The mean 2 is: ", af.mean(axis=1))
# print("The median is: ", np.median(af))
# print("The cumulative sum is: ", np.cumsum(af))
# print("The difference between two adjacent elements: ", np.diff(af))
# print("The indices of nonzero elements: ", np.nonzero(af))
# print("The nonzero elements: ", af[np.nonzero(af)])

bf = np.arange(14,2, -1).reshape((3,4))
print("New bf numbers\n", bf)
# print("The sorted bf:\n", np.sort(bf))
# print("bf.T:\n", bf.T)
print("clipped numbers: \n", np.clip(bf, 6, 10.1))
import csv
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import math
from sklearn.metrics import r2_score
import random
from numpy.linalg import inv

#Inhibitory Neuron
Ni =200
ri= random.uniform(Ni,  1)

#Excitatory Neuron
Ne = 800
re= random.uniform(Ne, 1)
print(re)

val = np.ones((Ne,1), dtype=int)

#a= np.matrix((0.02 * val) )
#b= np.matrix(0.2 * val)
#print (a)
#print (b)

zeors_array = np.zeros( (2, 3) )
print(zeors_array)

ones_array = np.ones( (1, 5), dtype=np.int32 )
print(ones_array)

a = np.array([1, 2, 3])
print(a)               # Output: [1, 2, 3]

A = np.array([[1, 2, 3], [3, 4, 5]])
print(A)

A = np.array([[2, 4], [5, -6]])
B = np.array([[9, -3], [3, 6]])


print("dodawanie")
C = A + B      # element wise addition
print(C)

print("mnozenie")
C = A.dot(B)
print(C)

print("macierz")
print(A)
print("macierz trannsponowa")
print(A.transpose())


print("macierz")
print(A)
print("macierz odwrotna")
print( np.linalg.inv(A))

print("macierz")
print(A)
print("wyzncznik macierzy")
print( np.linalg.det(A))

# 3x3 matrix
X =np.array([[12,7,3],
    [4 ,5,6],
    [7 ,8,9]])
# 3x4 matrix
Y =np.array([[5,8,1],
    [6,7,3],
    [4,5,9]])
print("mnozenie macierzy")
Z=np.dot(X,Y)
print(Z)


Xinv = inv(X)
print(" macierz odwrotna")

print(Xinv)

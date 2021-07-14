#============== Se importan Librerías ==============#
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import curve_fit

#============== Importación de datos ==============#

mat = scipy.io.loadmat('mediciones.mat')

#============== Definición de función ==============#

def Fun(d):
    return 22.7*np.log10(d)+41

#******************** PARTE 2.8M ********************# 
#============== Listas ==============#
path_loss_list = []
path_loss = []
R = mat["d28"]
Y = mat["pl28"]
x = []

#============== Calculos ==============#

for i in range(len(R)):
    lista = []
    PL = Fun(R[i])
    x.append(float(R[i]))
    lista.append(PL)
    path_loss.append(PL)
    path_loss_list.append(lista)
path_loss_list = np.array(path_loss_list)

#============== Calculo de error ==============#

error = []
for i in range(len(path_loss_list)):
    error.append(mse(path_loss_list[i],Y[i], squared=False))
   
error.sort()
error = np.array(error)
print("Error promedio: {}".format(np.mean(error)))
print("Mediana del error: {}".format((error[67]+error[68])/2))

#============== Graficos ==============#

x_line = np.arange(min(x), max(x), 1)
y_line = []
for i in x_line:
    y_line.append(Fun(i))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("F-LOS v2 2.8m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()

#******************** PARTE 4.7M ********************# 
#============== Listas ==============#
path_loss_list = []
path_loss = []
R = mat["d47"]
Y = mat["pl47"]
x = []

#============== Calculos ==============#

for i in range(len(R)):
    lista = []
    PL = Fun(R[i])
    x.append(float(R[i]))
    lista.append(PL)
    path_loss.append(PL)
    path_loss_list.append(lista)
path_loss_list = np.array(path_loss_list)

#============== Calculo de error ==============#

error = []
for i in range(len(path_loss_list)):
    error.append(mse(path_loss_list[i],Y[i], squared=False))
   
error.sort()
error = np.array(error)
print("Error promedio: {}".format(np.mean(error)))
print("Mediana del error: {}".format((error[67]+error[68])/2))

#============== Graficos ==============#

x_line = np.arange(min(x), max(x), 1)
y_line = []
for i in x_line:
    y_line.append(Fun(i))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("F-LOS v2 4.7m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()
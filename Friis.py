#============== Se importan Librerías ==============#
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import curve_fit

#============== Importación de datos ==============#

mat = scipy.io.loadmat('mediciones.mat')
fc = 58*10**8
luz = 3*10**8
lamb = luz/fc

#============== Definición de función ==============#

def PL28(x,lamb):
    return -10*np.log10((lamb**2)/(((4*np.pi)**2)*x*x))

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
    PL = PL28(R[i],lamb)
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
    y_line.append(PL28(i,lamb))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("Friis 2.8m")
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
    PL = PL28(R[i],lamb)
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
    y_line.append(PL28(i,lamb))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("Friis 4.7m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()

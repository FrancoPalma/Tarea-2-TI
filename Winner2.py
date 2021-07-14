#============== Se importan Librerías ==============#
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as mse

#============== Definición de función ==============#

def Fun(x):
    return 40*np.log10(x) + 9.45 - 17.3 * np.log10(2.8) - 17.3 * np.log10(1.6) + 2.7 * np.log10(fc/5.0)
fun = np.frompyfunc(Fun, 3, 1)

def fun2(x):
    return 40*np.log10(x) + 9.45 - 17.3 * np.log10(4.7) - 17.3 * np.log10(1.6) + 2.7 * np.log10(fc/5.0)

#============== Importación de datos ==============#

mat = scipy.io.loadmat('mediciones.mat')
fc = 5.8*(10**9)
luz = 3*(10**8)
l = luz/fc

#******************** PARTE 2.8M ********************# 
#============== Listas ==============#

path_loss_list = []
R = mat['d28']
Y = mat['pl28']
aux = []
auxY = []

#============== Calculos ==============#

for i in range(len(R)):
    lista = []
    a = float(R[i])
    aux.append(a)
    auxY.append(float(Y[i]))
    path_loss = Fun(a)
    lista.append(path_loss)
    path_loss_list.append(lista)
path_loss_list = np.array(path_loss_list)

#============== Calculo de error ==============#

print("Caso antena 2.8")
error_prom = []
rmse = []
for i in range(len(R)):
    rmse.append(mse(Fun(R[i]),Y[i], squared=False))
    error_prom.append(Fun(R[i])- Y[i])

error_prom.sort()
print("Error promedio: {}".format(np.mean(error_prom)))
print("RMSE: {}".format(np.mean(rmse)))
print("Mediana del error: {}".format((error_prom[67]+error_prom[68])/2))


#============== Graficos ==============#

x_line = np.arange(min(aux), max(aux), 1)
y_line = []
for i in x_line:
    y_line.append(Fun(i))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*np.log10(R[i])

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*np.log10(x_line[i])

plt.title("Winner2 2.8m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()
plt.clf()

#******************** PARTE 4.7M ********************# 
#============== Listas ==============#
print('\n',"="*20,'\n')
path_loss_list = []
R = mat["d47"]
Y = mat["pl47"]
aux = []

#============== Calculos ==============#

for i in range(len(R)):
    lista = []
    a = float(R[i])
    aux.append(a)
    path_loss = fun2(a)
    lista.append(path_loss)
    path_loss_list.append(lista)
path_loss_list = np.array(path_loss_list)

#============== Calculo de error ==============#

print("Caso antena 4.7")
error_prom = []
rmse = []
for i in range(len(R)):
    rmse.append(mse(Fun(R[i]),Y[i], squared=False))
    error_prom.append(Fun(R[i])- Y[i])

error_prom.sort()
print("Error promedio: {}".format(np.mean(error_prom)))
print("RMSE: {}".format(np.mean(rmse)))
print("Mediana del error: {}".format((error_prom[67]+error_prom[68])/2))


#============== Graficos ==============#

x_line = np.arange(min(aux), max(aux), 1)
y_line = []
for i in x_line:
    y_line.append(fun2(i))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*np.log10(R[i])

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*np.log10(x_line[i])

plt.title("Winner2 4.7m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()

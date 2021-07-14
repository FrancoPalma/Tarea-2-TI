#============== Se importan Librerías ==============#
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

#============== Definición de función ==============#

def fun(s,r,D,lamb):
    return 20*np.log10((np.exp(s*r)*4*np.pi*r*D)/lamb)

#============== Importación de datos ==============#

mat = scipy.io.loadmat('mediciones.mat')
fc = 5800000000
luz = 300000000
lamb = luz/fc

#******************** PARTE 2.8M ********************# 
#============== Listas ==============#

R = mat["d28"]
Y = mat["pl28"]
path_loss = []
x = []
y = []
error = []

#============== Calculos ==============#

h_t = 2.8
h_r = 1.6
h_0  = 1.0
r_bp = (4*(h_t-h_r)*(h_r-h_0))/lamb
s = 0.002


for i in range(len(R)):

    r = float(R[i])
    if(r <= r_bp):
        D = 1
    else:
        D = r/r_bp
    PL = fun(s,r,D,lamb)


    path_loss.append(PL)

    x.append(float(R[i]))

#============== Graficos ==============#

x_line = np.arange(min(x), max(x), 1)
y_line = []
for i in x_line:
    if(i <= r_bp):
        D = 1
    else:
        D = i/r_bp
    y_line.append(fun(s,i,D,lamb))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("F-Los 2.8m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()

#============== Calculo de error ==============#
for i in range(len(R)):
    error.append(path_loss[i] -float(Y[i]))

error.sort()   
error = np.array(error)
MSE = mse(Y, path_loss)
RMSE = math.sqrt(MSE)

print("Error promedio: ", np.mean(error))
print("Mediana: ", (error[67]+error[68])/2)
print("Error RMS",RMSE)

#******************** PARTE 4.7M ********************# 
#============== Listas ==============#

R = mat["d47"]
Y = mat["pl47"]
path_loss = []
x = []
error = []

#============== Calculos ==============#

h_t = 4.7
h_r = 1.6
h_0  = 1.0
r_bp = (4*(h_t-h_r)*(h_r-h_0))/lamb

for i in range(len(R)):
    aux = []
    r = float(R[i])
    if(r <= r_bp):
        D = 1
    else:
        D = r/r_bp
    PL = fun(s,r,D,lamb)
    aux.append(PL)
    aux = np.array(aux)

    path_loss.append(PL)
    x.append(float(R[i]))

#============== Graficos ==============#

x_line = np.arange(min(x), max(x), 1)
y_line = []
for i in x_line:
    if(i <= r_bp):
        D = 1
    else:
        D = i/r_bp
    y_line.append(fun(s,i,D,lamb))
fig, axes = plt.subplots()
for i in range(len(R)):
    R[i] = 10*math.log(R[i],10)

plt.scatter(R,Y)

for i in range(len(x_line)):
    x_line[i] = 10*math.log(x_line[i],10)

plt.title("F-Los 4.7m")
plt.plot(x_line, y_line, '-', color='red')
plt.xlabel("10Log(d[m])")
plt.ylabel("Path-losses[dB]")
plt.show()

#============== Calculo de error ==============#

for i in range(len(R)):
    error.append(path_loss[i] - float(Y[i]))

error.sort()   
error = np.array(error)
MSE = mse(Y, path_loss)
RMSE = math.sqrt(MSE)

print("Error promedio: ", np.mean(error))
print("Mediana: ", (error[67]+error[68])/2)
print("Error RMS",RMSE)

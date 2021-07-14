#============== Se importan Librerías ==============#
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as mse
import math

#============== Definición de funciones ==============#

def PL28(d,n1,n2):
    lamb = 300000000/5800000000
    d_bp = (4*1.8*0.6)/lamb
    if (d <= d_bp and d >= 1):
        return 20*np.log10((4*np.pi)/lamb) + 10*n1*np.log10(d/1)
    else:
        return 20*np.log10((4*np.pi)/lamb) + 10*n1*np.log10(d/1) + 10*n2*np.log10(d/d_bp)

def PL47(d,n1,n2):
    lamb = 300000000/5800000000
    d_bp = (4*3.7*0.6)/lamb
    if (d <= d_bp and d >= 1):
        return 20*np.log10((4*np.pi)/lamb) + 10*n1*np.log10(d/1)
    else:
        return 20*np.log10((4*np.pi)/lamb) + 10*n1*np.log10(d/1) + 10*n2*np.log10(d/d_bp)

def parametros(pl):
#============== Listas ==============#
    if(pl=='PL28'):
        x = mat['d28']
        y = mat['pl28']
        pl28=np.frompyfunc(PL28,3,1)
    elif(pl=='PL47'):
        x = mat['d47']
        y = mat['pl47']
        pl47=np.frompyfunc(PL47,3,1)

    x_peq = []
    x_grand = []
    y_peq = []
    y_grand = []

    for i in range(len(x)):
        if x[i] >= 1 and x[i] <= d_bp:
            x_peq.append(x[i])
            y_peq.append(y[i])
        else:
            x_grand.append(x[i])
            y_grand.append(y[i])

    n1 = 0
    n2 = 0

#============== Calculos ==============#

    while True:
        prom_real=0
        prom_emp=0
        for i in range(len(x)):
            if x[i] >= 1 and x[i] <= d_bp:
                if(pl == 'PL28'):
                    prom_real += PL28(x[i],n1,n2)
                    prom_emp += y[i]
                elif(pl == 'PL47'):
                    prom_real += PL47(x[i],n1,n2) 
                    prom_emp += y[i]
        prom_real = prom_real/len(x)
        prom_emp = prom_emp/len(y)
        if (mse(prom_real, prom_emp, squared=False) > 0.1):
            n1+=0.01
        else:
            break

    while True:
        prom_real=0
        prom_emp=0
        for i in range(len(x)):
            if x[i] >= d_bp:
                if(pl == 'PL28'):
                    prom_real += PL28(x[i],n1,n2)
                    prom_emp += y[i]
                elif(pl == 'PL47'):
                    prom_real += PL47(x[i],n1,n2)
                    prom_emp += y[i]
        prom_real = prom_real/len(x)
        prom_emp = prom_emp/len(y)
        if (mse(prom_real, prom_emp, squared=False) > 0.1):
            n2+=0.01
        else:
            break

#============== Calculo de error ==============#

    rmse = []
    error_prom = []
    if(pl == 'PL28'):
        for i in range(len(x)):
            rmse.append(mse(pl28(x[i], n1, n2),y[i], squared=False))
            error_prom.append(pl28(x[i],n1,n2) - y[i])
    elif(pl == 'PL47'):
        for i in range(len(x)):
            rmse.append(mse(pl47(x[i], n1, n2),y[i], squared=False))
            error_prom.append(pl47(x[i],n1,n2) - y[i])

    error_prom.sort()
    print("Error promedio: {}".format(np.mean(error_prom)))
    print("RMSE: {}".format(np.mean(rmse)))
    print("Mediana del error: {}".format((error_prom[67]+error_prom[68])/2))
    print("Valor de n1: {}".format(n1))
    print("Valor de n2: {}".format(n2))

#============== Graficos ==============#

    x_line = np.arange(min(x), max(x), 1)
    y_line = []
    for i in x_line:
        if pl == 'PL28':
            y_line.append(PL28(i, n1, n2))
        elif pl == 'PL47':
            y_line.append(PL47(i, n1, n2))
    for i in range(len(x_line)):
        x_line[i] = 10*math.log(x_line[i],10)
    plt.plot(x_line, y_line, '-', color='red')
    for i in range(len(x)):
        x[i] = 10*math.log(x[i],10)
    plt.scatter(x, y)

    plt.xlabel("10Log(d[m])")
    plt.ylabel("Path-losses[dB]")
    if pl == 'PL28':
        plt.title("Ajuste de 2 pendientes 2.8m")
        plt.savefig("PL28")
        plt.clf()
    elif pl == 'PL47':
        plt.title("Ajuste de 2 pendientes 4.7m")
        plt.savefig("PL47")
        plt.clf()

    return (n1,n2, np.mean(error_prom), (error_prom[67]+error_prom[68])/2 , np.mean(rmse))

#============== Importación de datos ==============#

mat = scipy.io.loadmat('mediciones.mat')

lamb = 300000000/5800000000
d_bp = (4*1.8*0.6)/lamb

#******************** PARTE 2.8M ********************# 
print("Caso antena 2.8 metros")
parametros('PL28')
print("-----------------------------------------------")

#******************** PARTE 4.7M ********************# 
print("Caso antena 4.7 metros")
parametros('PL47')


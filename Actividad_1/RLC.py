# %% [markdown]
# # Actividad 1
# ## Representacion de Sistemas y controladores
# 
# Se desarrollara la resolucion de los ejercicios planteados utilizando Python.
# Primero, vamos a importar librerias y paquetes a utilizar.
# Se debe ejecutar el siguiente código en una celda de código:

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
from control.matlab import *
from IPython.display import Image
from scipy import signal
from math import log



# %% [markdown]
# ### Circuito RLC
# 
# ![Circuito RLC](images/circuit_RLC.png)
# 
# 
# 
# 
# Sea el sistema eléctrico , con sus representacion en variables de estado : 
# 
# $\dot{x}=A x(t)+b u(t)$
# 
# $y=c^{T}x(t)$
# 
# Donde las matrices contienen a los coeficientes del circuito,
# 
# $ A =\begin{bmatrix}
#  -R/L & -1/L \\ 
#  1/C & 0 
# \end{bmatrix}$
# 
# $ B = \begin{bmatrix}
# 1/L \\ 
# 0
# \end{bmatrix} $
# 
# $ c^T =\begin{bmatrix}
#  R & 0
# \end{bmatrix} $

# %% [markdown]
# ### Item [1]
# 
# Asignar valores a R=47ohm, L=1uHy, y C=100nF. Obtener simulaciones que permitan
# estudiar la dinámica del sistema, con una entrada de tensión escalón de 12V, que cada 1ms cambia
# de signo.

# %% [markdown]
# Para la simulacion se va a utilizar una entrada de 12V la cual cambia de signo cada 1ms

# %%

t_sim = 1000  # Duración de la simulación en ms
t = np.linspace(0, 0.01, t_sim)  # Arreglo de tiempo en ms
frecuencia = 500  # Frecuencia de la señal en Hz

delay = 0.001

# Generar señal cuadrada con fase ajustada para que empiece en 0 V en t=0
entrada = 12 * signal.square(2 * np.pi * frecuencia * t-np.pi, duty=0.5)

# Ajustar la señal para que esté en 0 V en t = 0 ms
entrada[0] = 0

entrada_delayed= np.where(t >= delay, entrada, 0)

# Visualizar la señal de entrada
plt.figure(figsize=(10, 4))
plt.plot(t, entrada_delayed, drawstyle='steps-pre')
plt.title('Señal de entrada escalón')
plt.xlabel('Tiempo (S)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Dinamica del sistema
# 
# Estudiar dinamica del sistema, obteniendo la FDT y analizando sus polos.
# Para determinar el paso o tiempo de integracion se busca el polo que corrsponde a la dinamica mas rápida para la cual se llega a un 95%. El paso de integración debe ser al menos 10 veces mas chico que el timepo calculado
# 
# Para determinar el tiempode simulacion se busca el polo que corresponde con la dinámica mas lenta para la cual se llega a un 5%

# %%

R=47
L=1e-6
C=100e-9

A=[[-R/L, -1/L], [1/C, 0]]
B=[[1/L], [0]]
C=[[R,0]] 
D=[[0]]                 

# Obtener la función de transferencia
tf = ct.minreal(ss2tf(A, B, C, D))

# Imprimir la función de transferencia
print(tf)

#Obtener polos de la función de transferencia
poles = ct.poles(tf)
print(poles)    

#Paso de integracion (h) -> Dinamica rapida (polo mas alejado eje imaginario)

pR = min(poles)
print("El polo más rápido es: {:e}".format(pR))

h=((log(0.95)/pR)/10)
print("El paso de integracion es:{:e}".format(h))

#Tiempo de simulacion (ts)

pL=max(poles) 
print("El polo más rápido es: {:e}".format(pL))

ts=((log(0.05)/pL)*5)
print("El tiempo de simulacion es :{:e}".format(ts))


# %% [markdown]
# En primer lugar simularemos con los siguientes valores:
# 
# - $ R = 47  \Omega $
# - $ L = 1  \mu Hy $
# - $ C = 100  nF $
# 
# Luego simulamos la respuesta del circuito
# 

# %%
R=47
L=1e-6
C=100e-9

A=[[-R/L, -1/L], [1/C, 0]]
B=[[1/L], [0]]
C1=[[1, 0]] #Matriz para medir corriente
C2=[[0, 1]] #Matriz para medir voltage
D=[[0]]

sys1 = signal.StateSpace(A, B, C1, D) #voltaje capacitor
sys2 = signal.StateSpace(A, B, C2, D) #corrient

#------------------------------------------------#
t_sim = 200 # Duración de la simulación en ms
t = np.linspace(0, 0.01, t_sim)  # Arreglo de tiempo en ms
frecuencia = 500  # Frecuencia de la señal en Hz

delay = 0.001

# Generar señal cuadrada con fase ajustada para que empiece en 0 V en t=0
entrada = 12 * signal.square(2 * np.pi * frecuencia * t-np.pi, duty=0.5)

# Ajustar la señal para que esté en 0 V en t = 0 ms
entrada[0] = 0

u= entrada_delayed= np.where(t >= delay, entrada, 0)

#------------------------------------------------#

# Simular la respuesta del sistema
t1,y1,x1= signal.lsim(sys1,u, t) # simular sistemas lineales e invariantes en el tiempo (LTI)
t2,y2,x2= signal.lsim(sys2,u, t) 

# Visualizar la salida del sistema
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(t1, y1, 'b-', linewidth=2.5,label='Corriente')
plt.grid()
plt.title('Dinamica Sistema')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Corriente [A]')
plt.ylim(-0.1, 0.1)
plt.xlim(0, 0.01) 

plt.subplots_adjust(hspace = 0.5)  # Ajustar el espacio entre los subplots


plt.subplot(3, 1, 2)
plt.plot(t2, y2, 'r-', linewidth=2.5, label='Tension')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Tension[V]')
plt.grid()


plt.subplot(3, 1, 3)
plt.plot(t, u, 'g-', linewidth=2.5, label='Entrada')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Entrada[V]')
plt.grid()


# %% [markdown]
# Tomando como ejemplo otros valores vistos en clases:
# 
# - $ R = 4.7 k\Omega $
# - $ L = 10  \mu Hy $
# - $ C = 100  nF $
# 
# Luego simulamos la respuesta del circuito
# 

# %%
R=4.7e3
L=10e-6
C=100e-9

A=[[-R/L, -1/L], [1/C, 0]]
B=[[1/L], [0]]
C1=[[1, 0]] #Matriz para medir corriente
C2=[[0, 1]] #Matriz para medir voltage
D=[[0]]

sys1 = signal.StateSpace(A, B, C1, D) #voltaje capacitor
sys2 = signal.StateSpace(A, B, C2, D) #corrient

#------------------------------------------------#
t_sim = 1000 # Duración de la simulación en ms
t = np.linspace(0, 0.01, t_sim)  # Arreglo de tiempo en ms
frecuencia = 500  # Frecuencia de la señal en Hz

delay = 0.001

# Generar señal cuadrada con fase ajustada para que empiece en 0 V en t=0
entrada = 12 * signal.square(2 * np.pi * frecuencia * t-np.pi, duty=0.5)

# Ajustar la señal para que esté en 0 V en t = 0 ms
entrada[0] = 0

u= entrada_delayed= np.where(t >= delay, entrada, 0)

#------------------------------------------------#

# Simular la respuesta del sistema
t1,y1,x1= signal.lsim(sys1,u, t) # simular sistemas lineales e invariantes en el tiempo (LTI)
t2,y2,x2= signal.lsim(sys2,u, t) 

# Visualizar la salida del sistema
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(t1, y1, 'b-', linewidth=2.5,label='Corriente')
plt.grid()
plt.title('Dinamica Sistema')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Corriente [A]')

plt.subplots_adjust(hspace = 0.5)  # Ajustar el espacio entre los subplots


plt.subplot(3, 1, 2)
plt.plot(t2, y2, 'r-', linewidth=2.5, label='Tension')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Tension[V]')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, u, 'g-', linewidth=2.5, label='Entrada')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Entrada[V]')
plt.grid()








# %% [markdown]
# ### Item [2] - Determinar valores componentes
# En el archivo Curvas_Medidas_RLC.xls (datos en la hoja 1 y etiquetas en la hoja 2)
# están las series de datos que sirven para deducir los valores de R, L y C del circuito. Emplear el
# método de la respuesta al escalón, tomando como salida la tensión en el capacitor.
# 

# %% [markdown]
# En primer lugar a partir de los datos extraidos graficaremos las curvas de tension de entrada y corriente y tension en el capacitor. Para ello extraemos datos del archivo .xls

# %%


df= pd.read_excel('Curvas_Medidas_RLC_2024.xls') # extraigo datos de xls

t = df.iloc[:, 0] #selecciono primera columna y todas sus filas, guardo como variable t
ic_t = df.iloc[:, 1]
Vc_t = df.iloc[:, 2]
Vin_t= df.iloc[:, 3]

## Graficos de variables
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(t,ic_t,'b-', label='ic_t')
plt.title('Dinámica del sistema')
plt.xlabel('Tiempo (t)')
plt.ylabel('ic_t')
plt.legend()
plt.grid(True)


plt.show()

plt.subplots_adjust(hspace = 0.5)  # Ajustar el espacio entre los subplot

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 2)
plt.plot(t,Vc_t ,'r-', label='Vc_t')
plt.xlabel('Tiempo (t)')
plt.ylabel('Vc_t')
plt.legend()
plt.grid(True)

mplcursors.cursor(hover=True)
plt.show()

plt.subplots_adjust(hspace = 0.5)  # Ajustar el espacio entre los subplot

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 3)
plt.plot(t,Vin_t ,'g-', label='Vin_t')
plt.xlabel('Tiempo (t)')
plt.ylabel('Vinc_t')
plt.legend()
plt.grid(True)

mplcursors.cursor(hover=True)
plt.show()




# %% [markdown]
# El objetivo ahora, es a partir de dichos graficos es obtener la funcion de transferencia del sistema, la que despues nos permite, en nuestro caso , obtener los valores de R,L,C del circuito
# 
# Basandonos en las variables de entrada y salida
# 
# Partiendo nuevamente del circuito
# 
# ![Circuito RLC](images/circuit_RLC.png)
# 
# Planteando las ecuaciones diferenciales del mismo
# 
# 
# $\frac{di}{dt}=-\frac{R}{L}i-\frac{1}{L}vc+\frac{1}{L}ve$
# 
# $\frac{dv_c}{dt}=-\frac{1}{C}i$
# 
# Podemos expresar las mismas en una ecuacion matricial-vectorial
# 
# $\begin{bmatrix}\frac{di}{dt}\\ \frac{dv_c}{dt}\end{bmatrix} $ 
# $ = \begin{bmatrix}-R/L &-1/L \\ 1/C & 0 \end{bmatrix} $
# $\begin{bmatrix}i\\vc \end{bmatrix} $
# $+ \begin{bmatrix}1/L\\0 \end{bmatrix}$
# $\begin{bmatrix}V_e \end{bmatrix}$
# 
# Definiendo a i,vc como variables de estado y a x como vector de estado , podemos expresarlo como:
# 
# $\dot{x}=A x(t)+b(u)$
# 
# Transformando al dominio de laplace el conjunto de  ecuaciones
# 
# $sI(s)=\frac{1}{L}$
# 
# $sV_c(s)=\frac{1}{C}I(s)$
# 
# Despejamos I de ambas ecuaciones para igualar y obtener la FdT de la tension en el capcitor
# 
# $G(s)=\frac{1}{LCs^{2}-CRs+1}$
# 
# Se obvserva que es una FdT de segundo grado y tiene dos polos reales y distintos. Aplicaremos método Chen para reconocerla.
# 
# $G(s)=\frac{K(T_3s+1)}{(T_1s+1)(T_2s+1)}$
# 
# Basandonos en los recursos de Identificacion.ipynb brindados en clase
# 
# Como nos indica el metodo debemos definir un intervalo de tiempo $t_1$ que será usado de referencia, que es lo que se nos pide en el metodo de de chen.
# 

# %% [markdown]
import plotly.graph_objects as go
import pandas as pd

df= pd.read_excel('Curvas_Medidas_RLC_2024.xls') # extraigo datos de xls

t = df.iloc[:, 0] #selecciono primera columna y todas sus filas, guardo como variable t
ic_t = df.iloc[:, 1]
Vc_t = df.iloc[:, 2]
Vin_t= df.iloc[:, 3]

# Gráfico para ic_t
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=t,
    y=ic_t,
    mode='lines',
    name='ic_t',
    hovertemplate='Tiempo: %{x:.2f}s, Corriente: %{y:.2f}A'
))

fig1.update_layout(
    title='Dinámica del sistema',
    xaxis_title='Tiempo (t)',
    yaxis_title='ic_t',
)

fig1.show()

# Gráfico para Vc_t
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=t,
    y=Vc_t,
    mode='lines',
    name='Vc_t',
    hovertemplate='Tiempo: %{x:.2f}s, Vc_t: %{y:.2f}'
))

fig2.update_layout(
    title='Dinámica del sistema',
    xaxis_title='Tiempo (t)',
    yaxis_title='Vc_t',
)

fig2.show()

# Gráfico para Vin_t
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=t,
    y=Vin_t,
    mode='lines',
    name='Vin_t',
    hovertemplate='Tiempo: %{x:.2f}s, Vin_t: %{y:.2f}'
))

fig3.update_layout(
    title='Dinámica del sistema',
    xaxis_title='Tiempo (t)',
    yaxis_title='Vin_t',
)

fig3.show()
# %%

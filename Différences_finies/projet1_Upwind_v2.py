''' Import Module'''
import numpy as np
import matplotlib.pyplot as plt


dx = 0.1739
N = int(40/dx)+1
x = np.arange(-20,20,dx)
dt = 0.001
Nt = int(20/dt)
t = np.arange(0,20,dt)
delta = 0.022

A = 1
k = (A/2)**0.5
w = 4*k**3

un = np.zeros((Nt, N))
un0 = A/(np.cosh(k*x)**2)
un[0] = un0

# Upwind
for j in range(Nt-1):
    for i in range(0,N-2):
        if un[j][i]>0:
            un[j+1][i] = un[j][i] - (dt/dx)*un[j][i]*(un[j][i]-un[j][i-1]) - ((dt*delta**2)/(2*dx**3))*(un[j][i+2]-2*un[j][i+1]+2*un[j][i-1]-un[j][i-2])
        else:
            un[j+1][i] = un[j][i] - (dt/dx)*un[j][i]*(un[j][i+1]-un[j][i]) - ((dt*delta**2)/(2*dx**3))*(un[j][i+2]-2*un[j][i+1]+2*un[j][i-1]-un[j][i-2])
    if un[j][N-2]>0:
        un[j+1][N-2] = un[j][N-2] - (dt/dx)*un[j][N-2]*(un[j][N-2]-un[j][N-3]) - ((dt*delta**2)/(2*dx**3))*(un[j][0]-2*un[j][N-1]+2*un[j][N-3]-un[j][N-4])
    else:
        un[j+1][N-2] = un[j][N-2] - (dt/dx)*un[j][N-2]*(un[j][N-1]-un[j][N-2]) - ((dt*delta**2)/(2*dx**3))*(un[j][0]-2*un[j][N-1]+2*un[j][N-3]-un[j][N-4])
    if un[j][N-1]>0:
        un[j+1][N-1] = un[j][N-1] - (dt/dx)*un[j][N-1]*(un[j][N-1]-un[j][N-2]) - ((dt*delta**2)/(2*dx**3))*(un[j][1]-2*un[j][0]+2*un[j][N-2]-un[j][N-3])
    else:
        un[j+1][N-1] = un[j][N-1] - (dt/dx)*un[j][N-1]*(un[j][0]-un[j][N-1]) - ((dt*delta**2)/(2*dx**3))*(un[j][1]-2*un[j][0]+2*un[j][N-2]-un[j][N-3])

time = 0
plt.plot(x,un[time],label='t = 0')

time = int(Nt*0.25/(1))
plt.plot(x,un[time],label='t = 0.25')

time = int(Nt*0.5/(1))
plt.plot(x,un[time],label='t = 0.5')

time = int(Nt/(1)-1)
plt.plot(x,un[time],label='t = 1')
plt.legend()
plt.show()


for n in range(0,Nt,int(Nt/10)):
    if n==0: fig, ax = plt.subplots(figsize=(5.5,4))
    plt.clf()
    plt.plot(x,un[n,:])
    plt.gca()
    plt.title('Simulation')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.18)
    plt.draw()
    plt.pause(0.001)
plt.show()


[xx,tt]=np.meshgrid(x,t)
plt.contourf(xx,tt,un, cmap = 'jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.show()
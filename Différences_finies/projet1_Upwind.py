''' Import Module'''
import numpy as np
import matplotlib.pyplot as plt


dx = 0.1
N = int(2*np.pi/dx)+1
x = np.arange(0,2*np.pi,dx)
dt = 0.05
Nt = int(5/dt)
t = np.arange(0,5,dt)

c = 1*2*np.pi/10
mu0 = 2*np.pi/5
s0 = 2*np.pi/20
delta = 0.022

uexact = np.zeros((Nt, N))
for j in range(Nt):
    uexact[j] = np.exp(-((x-c*t[j]-mu0)**2)/(2*s0*s0)) + 0.5*np.exp(-((x-c*t[j]-3*mu0)**2)/(2*s0*s0))

un = np.zeros((Nt, N))
un0 = np.exp(-((x-mu0)**2)/(2*s0*s0)) + 0.5*np.exp(-((x-3*mu0)**2)/(2*s0*s0))
un[0] = un0

# upwind
for j in range(Nt-1):
    for i in range(0,N-2):
        if un[j][i]>0:
            un[j+1][i] = un[j][i] - (dt/dx)*un[j][i]*(un[j][i]-un[j][i-1]) - ((dt*delta**2)/(2*dx**3))*(un[j][i+2]-2*un[j][i+1]+2*un[j][i-1]-un[j][i-2])
        else:
            un[j+1][i] = un[j][i] - (dt/dx)*un[j][i]*(un[j][i+1]-un[j][i]) - ((dt*delta**2)/(2*dx**3))*(un[j][i+2]-2*un[j][i+1]+2*un[j][i-1]-un[j][i-2])
    if un[j][N-2]>0:
        un[j+1][N-2] = un[j][N-2] - (dt/dx)*un[j][N-2]*(un[j][N-2]-un[j][N-3]) - ((dt*delta*delta)/(2*dx**3))*(un[j][0]-2*un[j][N-1]+2*un[j][N-3]-un[j][N-4])
    else:
        un[j+1][N-2] = un[j][N-2] - (dt/dx)*un[j][N-2]*(un[j][N-1]-un[j][N-2]) - ((dt*delta*delta)/(2*dx**3))*(un[j][0]-2*un[j][N-1]+2*un[j][N-3]-un[j][N-4])
    if un[j][N-1]>0:
        un[j+1][N-1] = un[j][N-1] - (dt/dx)*un[j][N-1]*(un[j][N-1]-un[j][N-2]) - ((dt*delta*delta)/(2*dx**3))*(un[j][1]-2*un[j][0]+2*un[j][N-2]-un[j][N-3])
    else:
        un[j+1][N-1] = un[j][N-1] - (dt/dx)*un[j][N-1]*(un[j][0]-un[j][N-1]) - ((dt*delta*delta)/(2*dx**3))*(un[j][1]-2*un[j][0]+2*un[j][N-2]-un[j][N-3])

time = 1
plt.plot(x,uexact[time],label='sol exacte')
plt.plot(x,un[time],label='sol num')
plt.legend()
plt.show()


for n in range(0,Nt,int(Nt/100)):
    
    if n==0: fig, ax = plt.subplots(figsize=(5.5,4))
    plt.clf()
    plt.plot(x,un[n,:])
    plt.scatter(x,uexact[n,:])
    plt.gca().legend(('numerique','exact'))
    plt.gca()
    plt.title('coucou')
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
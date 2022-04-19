''' Import Module'''
import numpy as np
import matplotlib.pyplot as plt


dx = 0.1739
N = int(2*np.pi/dx)+1
x = np.arange(-np.pi,np.pi,dx)
dt = 0.0002
Nt = int(20/dt)
t = np.arange(0,20,dt)

c = 1*2*np.pi/10
mu0 = 2*np.pi/5
s0 = 2*np.pi/20
delta = 0.022

A = 1
k = (A/2)**0.5
w = 4*k**3

uexact = np.zeros((Nt, N))
for j in range(Nt):
    #uexact[j] = np.exp(-((x-c*t[j]-mu0)**2)/(2*s0*s0))
    uexact[j] = A/(np.cosh(k*x-w*t[j])**2)

un = np.zeros((Nt, N))
#un0 = np.exp(-((x-mu0)**2)/(2*s0*s0)) + 2*np.exp(-((x-3*mu0)**2)/(2*s0*s0))
#un0 = 4*np.exp(-((x-mu0)**2)/(2*s0*s0))
un0 = A/(np.cosh(k*x)**2)
un[0] = un0

# leap-frog (Zabusky et Kruskal)
un[1] = un0
for i in range(0,N-2):
    un[1][i] = un[0][i] - (dt/(3*dx))*(un[0][i+1]+un[0][i]+un[0][i-1])*(un[0][i+1]-un[0][i-1]) - (dt*delta*delta/(2*dx**3))*(un[0][i+2]-2*un[0][i+1]+2*un[0][i-1]-un[0][i-2])
un[1][N-2] = un[0][N-2] - (dt/(3*dx))*(un[1][N-1]+un[1][N-2]+un[1][N-3])*(un[1][N-1]-un[1][N-3]) - ((dt*delta*delta)/(dx**3))*(un[1][0]-2*un[1][N-1]+2*un[1][N-3]-un[1][N-4])
un[1][N-1] = un[0][N-1] - (dt/(3*dx))*(un[1][0]+un[1][N-1]+un[1][N-2])*(un[1][0]-un[1][N-2]) - ((dt*delta*delta)/(dx**3))*(un[1][1]-2*un[1][0]+2*un[1][N-2]-un[1][N-3])

for j in range(1,Nt-1):
    for i in range(0,N-2):
        un[j+1][i] = un[j-1][i] - (dt/(3*dx))*(un[j][i+1]+un[j][i]+un[j][i-1])*(un[j][i+1]-un[j][i-1]) - ((dt*delta*delta)/(dx**3))*(un[j][i+2]-2*un[j][i+1]+2*un[j][i-1]-un[j][i-2])
    un[j+1][N-2] = un[j-1][N-2] - (dt/(3*dx))*(un[j][N-1]+un[j][N-2]+un[j][N-3])*(un[j][N-1]-un[j][N-3]) - ((dt*delta*delta)/(dx**3))*(un[j][0]-2*un[j][N-1]+2*un[j][N-3]-un[j][N-4])
    un[j+1][N-1] = un[j-1][N-1] - (dt/(3*dx))*(un[j][0]+un[j][N-1]+un[j][N-2])*(un[j][0]-un[j][N-2]) - ((dt*delta*delta)/(dx**3))*(un[j][1]-2*un[j][0]+2*un[j][N-2]-un[j][N-3])

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

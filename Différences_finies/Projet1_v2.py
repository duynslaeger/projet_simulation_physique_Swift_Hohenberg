''' Importation des modules '''
import numpy as np
import matplotlib.pyplot as plt


''' Fonctions utiles '''
#################### LEAPFROG ####################


def first_iter(un, Nt, N, dt, dx, delta):
    for i in range(0, N - 2):
        un[1][i] = un[0][i] - (dt / dx) * (un[0][i + 1] + un[0][i] + un[0][i - 1]) * (un[0][i + 1] - un[0][i - 1]) - (
            dt / (2 * dx**3)) * (un[0][i + 2] - 2 * un[0][i + 1] + 2 * un[0][i - 1] - un[0][i - 2])
    un[1][N - 2] = un[0][N - 2] - (dt / dx) * (un[0][N - 1] + un[0][N - 2] + un[0][N - 3]) * (
        un[0][N - 1] - un[0][N - 3]) - ((dt) / (2 * dx**3)) * (un[0][0] - 2 * un[0][N - 1] + 2 * un[0][N - 3] - un[0][N - 4])
    un[1][N - 1] = un[0][N - 1] - (dt / dx) * (un[0][0] + un[0][N - 1] + un[0][N - 2]) * (
        un[0][0] - un[0][N - 2]) - ((dt) / (2 * dx**3)) * (un[0][1] - 2 * un[0][0] + 2 * un[0][N - 2] - un[0][N - 3])
    return un


def compute(un, Nt, N, dt, dx, delta):
    for j in range(1, Nt - 1):
        for i in range(0, N - 2):
            un[j + 1][i] = un[j - 1][i] - (dt / (3 * dx)) * (un[j][i + 1] + un[j][i] + un[j][i - 1]) * (un[j][i + 1] - un[j][i - 1]) - (
                (dt * delta**2) / (dx**3)) * (un[j][i + 2] - 2 * un[j][i + 1] + 2 * un[j][i - 1] - un[j][i - 2])
        un[j + 1][N - 2] = un[j - 1][N - 2] - (dt / (3 * dx)) * (un[j][N - 1] + un[j][N - 2] + un[j][N - 3]) * (
            un[j][N - 1] - un[j][N - 3]) - ((dt * delta**2) / (dx**3)) * (un[j][0] - 2 * un[j][N - 1] + 2 * un[j][N - 3] - un[j][N - 4])
        un[j + 1][N - 1] = un[j - 1][N - 1] - (dt / (3 * dx)) * (un[j][0] + un[j][N - 1] + un[j][N - 2]) * (
            un[j][0] - un[j][N - 2]) - ((dt * delta**2) / (dx**3)) * (un[j][1] - 2 * un[j][0] + 2 * un[j][N - 2] - un[j][N - 3])
    return un


''' Condition initiale: cosinus '''
#################### LEAPFROG ####################
"""
dx = 0.01
N = int(2/dx)
x = np.arange(0,2,dx)
dt = 0.0001
Nt = int(2/dt)
t = np.arange(0,2,dt)
delta = 0.022

un = np.zeros((Nt, N))
un0 = np.cos(np.pi*x)
un[0] = un0
un[1] = un0

un = first_iter(un,Nt,N,dt,dx,delta)
un = compute(un,Nt,N,dt,dx,delta)

time = 0
plt.plot(x,un[time],label='t = 0')

time = int(1/(dt*np.pi))
plt.plot(x,un[time],label='t = 1/$\pi$')

time = int(3.6/(dt*np.pi))
plt.plot(x,un[time],label='t = 3.6/$\pi$')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x,t)$")
plt.show()
"""
"""
for n in range(0,Nt,int(Nt/10)):
    if n==0: fig, ax = plt.subplots(figsize=(5.5,4))
    plt.clf()
    plt.plot(x,un[n,:])
    plt.gca()
    plt.title('coucou')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.18)
    plt.draw()
    plt.pause(0.001)
plt.show()
"""

''' Condition initiale: sech^2 (1 soliton) '''
#################### LEAPFROG ####################
"""
''' Amplitude = 1 '''
dx = 0.1739
N = int(40/dx)+1
x = np.arange(-20,20,dx)
dt = 0.002
Nt = int(2/dt)
t = np.arange(0,2,dt)
delta = 0.4

A = 1
k = (A/2)**0.5
w = 4*k**3

un = np.zeros((Nt, N))
un0 = A/(np.cosh(k*x)**2)
un[0] = un0
un[1] = un0

un = first_iter(un,Nt,N,dt,dx,delta)
un = compute(un,Nt,N,dt,dx,delta)

plt.subplot(311)
time = 0
plt.plot(x,un[time],label='t = 0')

time = int(Nt*0.25/(1))
plt.plot(x,un[time],label='t = 0.25')

time = int(Nt*0.5/(1))
plt.plot(x,un[time],label='t = 0.5')

time = int(Nt/(1)-1)
plt.plot(x,un[time],label='t = 1')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x,t)$")


''' Amplitude = 2 '''
dx = 0.08
N = int(40/dx)
x = np.arange(-20,20,dx)
dt = 0.00019
Nt = int(2/dt)
t = np.arange(0,2,dt)
delta = 0.4

A = 2
k = (A/2)**0.5
w = 4*k**3

un = np.zeros((Nt, N))
un0 = A/(np.cosh(k*x)**2)
un[0] = un0
un[1] = un0

un = first_iter(un,Nt,N,dt,dx,delta)
un = compute(un,Nt,N,dt,dx,delta)

plt.subplot(312)
time = 0
plt.plot(x,un[time],label='t = 0')

time = int(Nt*0.25/(1))
plt.plot(x,un[time],label='t = 0.25')

time = int(Nt*0.5/(1))
plt.plot(x,un[time],label='t = 0.5')

time = int(Nt/(1)-1)
plt.plot(x,un[time],label='t = 1')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x,t)$")


''' Amplitude = 4 '''
dx = 0.05
N = int(40/dx)
x = np.arange(-20,20,dx)
dt = 0.00004
Nt = int(2/dt)
t = np.arange(0,2,dt)
delta = 0.4

A = 4
k = (A/2)**0.5
w = 4*k**3

un = np.zeros((Nt, N))
un0 = A/(np.cosh(k*x)**2)
un[0] = un0
un[1] = un0

un = first_iter(un,Nt,N,dt,dx,delta)
un = compute(un,Nt,N,dt,dx,delta)

plt.subplot(313)
time = 0
plt.plot(x,un[time],label='t = 0')

time = int(Nt*0.25/(1))
plt.plot(x,un[time],label='t = 0.25')

time = int(Nt*0.5/(1))
plt.plot(x,un[time],label='t = 0.5')

time = int(Nt/(1)-1)
plt.plot(x,un[time],label='t = 1')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x,t)$")

plt.show()
"""
"""
for n in range(0,Nt,int(Nt/10)):
    if n==0: fig, ax = plt.subplots(figsize=(5.5,4))
    plt.clf()
    plt.plot(x,un[n,:])
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
"""


''' Condition initiale: sech^2 (2 solitons) '''
#################### LEAPFROG ####################
dx = 0.12
N = int(40 / dx) + 1
x = np.arange(-20, 20, dx)
dt = 0.00066
Nt = int(20 / dt) + 1
t = np.arange(0, 20, dt)
delta = 0.4

A1 = 4
k1 = (A1 / 2)**0.5
w1 = 4 * k1**3
A2 = 1
k2 = (A2 / 2)**0.5
w2 = 4 * k2**3

mu0 = 5

un = np.zeros((Nt, N))
un0 = A1 / (np.cosh(k1 * x + 3 * mu0)**2) + A2 / (np.cosh(k2 * (x - mu0))**2)
un[0] = un0
un[1] = un0

un = first_iter(un, Nt, N, dt, dx, delta)
un = compute(un, Nt, N, dt, dx, delta)

time = 0
plt.plot(x, un[time], label='t = 0')

time = int(Nt * 0.25 / (1))
plt.plot(x, un[time], label='t = 0.25')

time = int(Nt * 0.5 / (1))
plt.plot(x, un[time], label='t = 0.5')

time = int(Nt / (1) - 1)
plt.plot(x, un[time], label='t = 1')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x,t)$")

plt.show()

for n in range(0, Nt, int(Nt / 65)):
    if n == 0:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    plt.clf()
    plt.plot(x, un[n, :])
    plt.gca()
    plt.title('Interaction entre deux solitons')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.ylim(0, 4.5)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.18)
    plt.draw()
    plt.pause(0.001)
plt.show()

[xx, tt] = np.meshgrid(x, t)
plt.contourf(xx, tt, un, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.show()

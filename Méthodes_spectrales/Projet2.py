""" Import """
import numpy as np
import matplotlib.pyplot as plt




N = 1024
r = 0.2
L = 100
x = np.linspace(0,L-L/N,N)
dt = 0.05
Nt = int(200/dt)
t = np.linspace(0,200,Nt)

un0 = np.cos(2*np.pi*x/L) + 0.1*np.cos(4*np.pi*x/L)
k = np.linspace(-N/2,N/2-1,N)

### DFT ###
"""
uk0 = np.fft.fftshift(np.fft.fft(un0))/N
plt.plot(k,uk0.real,label='real')
plt.plot(k,uk0.imag,label='imag')
plt.xlim(-5,5)
plt.legend()
plt.show()
"""

### ###
uk0 = np.fft.fftshift(np.fft.fft(un0))/N
uk = np.zeros((Nt, N),dtype=complex)
un = np.zeros((Nt, N))
uk[0] = uk0
un[0] = un0

fl = r - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)

Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[0]))).real)**3))
uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))*N).real

for i in range(2,Nt):
    Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-1]))*N).real)**3))/(N**3)
    FuAv = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-2]))*N).real)**3))/(N**3)
    uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] + dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
    un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))*N).real

[xx,tt]=np.meshgrid(x,t)
plt.contourf(xx,tt,un, cmap = 'jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.show()

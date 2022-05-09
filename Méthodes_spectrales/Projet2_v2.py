""" Import """
import numpy as np
import matplotlib.pyplot as plt


#############################################################

N = 1024
r = 0.2
L = 100
x = np.linspace(0,L-L/N,N)
dt = 0.05
Nt = int(200/dt)
t = np.linspace(0,200,Nt)

un0 = np.cos(2*np.pi*x/L) + 0.1*np.cos(4*np.pi*x/L)
uk0 = np.fft.fftshift(np.fft.fft(un0))

k = np.linspace(-N/2,N/2-1,N)

uk = np.zeros((Nt, N),dtype=complex)
un = np.zeros((Nt, N))
uk[0] = uk0
un[0] = un0

fl = r - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)

#Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[0]))).real)**3))
Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real

for i in range(2,Nt):
    #Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-1]))).real)**3))
    #FuAv = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-2]))).real)**3))
    Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
    FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
    uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
    un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real

c = np.linspace(np.min(un[-1]),np.max(un[-1]),101)
#c = np.linspace(-1,1,101)
[xx,tt]=np.meshgrid(x,t)
plt.contourf(xx,tt,un, c, cmap = 'jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title("r = {}, L = {}".format(r,L))
plt.show()
"""
"""
plt.plot(x,un[-5])
plt.title("r = {}, L = {}".format(r,L))
plt.show()
"""

### DFT ###
"""
uk0 = np.fft.fftshift(np.fft.fft(un0))/N
plt.plot(k,uk0.real,label='real')
plt.plot(k,uk0.imag,label='imag')
plt.xlim(-5,5)
plt.legend()
plt.show()






#############################################################

# N = 1024
# L = 100
# x = np.linspace(0,L-L/N,N)
# dt = 0.05
# k = np.linspace(-N/2,N/2-1,N)
# un0 = np.cos(2*np.pi*x/L) + 0.1*np.cos(4*np.pi*x/L)
# uk0 = np.fft.fftshift(np.fft.fft(un0))
# A = []
# r = np.arange(-0.2, 0.82, 0.02)

# for a in range(len(r)):
#     if r[a] <= 0:
#         Nt = int(200/dt)
#         t = np.linspace(0,200,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0<r[a]<0.12:
#         Nt = int(400/dt)
#         t = np.linspace(0,400,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0.1<r[a]<0.22:
#         Nt = int(300/dt)
#         t = np.linspace(0,300,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0.2<r[a]<0.32:
#         Nt = int(200/dt)
#         t = np.linspace(0,200,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0.3<r[a]<0.62:
#         Nt = int(100/dt)
#         t = np.linspace(0,100,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0.6<r[a]<0.72:
#         Nt = int(75/dt)
#         t = np.linspace(0,75,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 0.7<r[a]<0.82:
#         Nt = int(50/dt)
#         t = np.linspace(0,50,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
# """
#     elif 0.8<r[a]<1.72:
#         Nt = int(40/dt)
#         t = np.linspace(0,40,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 1.7<r[a]<3.02:
#         Nt = int(20/dt)
#         t = np.linspace(0,20,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
#     elif 3.0<r[a]<5.02:
#         Nt = int(15/dt)
#         t = np.linspace(0,15,Nt)
#         uk = np.zeros((Nt, N),dtype=complex)
#         un = np.zeros((Nt, N))
#         uk[0] = uk0
#         un[0] = un0
#         fl = r[a] - 1 + 2*((2*np.pi*k/L)**2) - ((2*np.pi*k/L)**4)
#         Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
#         uk[1] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[0] + dt*((Fu)/(1-0.5*dt*fl))
#         un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real
#         for i in range(2,Nt):
#             Fu = np.fft.fftshift(np.fft.fft(un[i-1]**3))
#             FuAv = np.fft.fftshift(np.fft.fft(un[i-2]**3))
#             uk[i] = ((1+0.5*dt*fl)/(1-0.5*dt*fl))*uk[i-1] - dt*((1.5*Fu-0.5*FuAv)/(1-0.5*dt*fl))
#             un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
#         A.append((np.mean(un[-1]**2))**0.5)
# """
# plt.plot(r,A)



# r2 = []
# A2 = []
# for i in range(len(r)):
#     if r[i]>0:
#         r2.append(r[i])
#         A2.append(A[i])

# from scipy.optimize import curve_fit
# def func(x, a, b, c):
#     return a * ((x)**b) + c
# popt, pcov = curve_fit(func, r2, A2)
# plt.plot(r2, func(r2, *popt), 'r-', label='fit: $%5.3f \cdot r^{%5.3f} {%5.3f}$' % tuple(popt))
# plt.legend()

# plt.xlabel('r')
# plt.ylabel('A')
# plt.show()

''' Import Module'''
import numpy as np
import matplotlib.pyplot as plt

u0 = 1
k = 0.1
h = 0.2
delta = 0.022

alpha = u0*k/h
beta = k*delta**2/(2*h**3)

r = np.linspace(0,2*np.pi,101)
h2 = 1
z = 1 - alpha*(1-np.cos(r*h2)+1j*np.sin(r*h2))
z1 = z - beta*(4j*np.sin(r*h2)*(np.cos(r*h2)-1))

print(alpha)
print(beta)

#plot unit circle
t = np.linspace(0,2*np.pi,101)
plt.plot(np.cos(t),np.sin(t),label="Unit circle")

#plot data
plt.plot(z.real, z.imag,label="Two first terms")
plt.plot(z1.real, z1.imag,label="Three terms")

#draw horizontal and vertical axes
m = max(max(abs(z.real)),max(abs(z.imag)))
plt.xlim(-1.1*m,1.1*m),plt.ylim(-1.1*m,1.1*m)
plt.plot([0,0],[-1.1*m,1.1*m],'k',linewidth=0.5)
plt.plot([-1.1*m,1.1*m],[0,0],'k',linewidth=0.5)
plt.axis("equal")
plt.legend()
plt.show()
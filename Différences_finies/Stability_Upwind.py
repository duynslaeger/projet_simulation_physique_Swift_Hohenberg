''' Importation des modules '''
import numpy as np
import matplotlib.pyplot as plt


#################### Cercle complexe ####################
''' plot données '''
u0 = 1
k = 0.01
h = 0.1
delta = 0.022
alpha = u0*k/h
beta = k*delta**2/(2*h**3)
#print(alpha)
#print(beta)
r = np.linspace(0,2*np.pi,101)
h2 = 1
z = 1 - alpha*(1-np.cos(r*h2)+1j*np.sin(r*h2))
z1 = z - beta*(4j*np.sin(r*h2)*(np.cos(r*h2)-1))
plt.plot(z.real, z.imag,label="Deux premiers termes (k = {}, h = {})".format(k,h), color="blue")
plt.plot(z1.real, z1.imag, '--', label="Tous les termes (k = {}, h = {})".format(k,h), color="blue")

u0 = 1
k = 0.1
h = 0.1
delta = 0.022
alpha = u0*k/h
beta = k*delta**2/(2*h**3)
#print(alpha)
#print(beta)
r = np.linspace(0,2*np.pi,101)
h2 = 1
z = 1 - alpha*(1-np.cos(r*h2)+1j*np.sin(r*h2))
z1 = z - beta*(4j*np.sin(r*h2)*(np.cos(r*h2)-1))
plt.plot(z.real, z.imag,label="Deux premiers termes (k = {}, h = {})".format(k,h), color="green")
plt.plot(z1.real, z1.imag, '--', label="Tous les termes (k = {}, h = {})".format(k,h), color="green")

u0 = 1
k = 0.1
h = 0.05
delta = 0.022
alpha = u0*k/h
beta = k*delta**2/(2*h**3)
#print(alpha)
#print(beta)
r = np.linspace(0,2*np.pi,101)
h2 = 1
z = 1 - alpha*(1-np.cos(r*h2)+1j*np.sin(r*h2))
z1 = z - beta*(4j*np.sin(r*h2)*(np.cos(r*h2)-1))
plt.plot(z.real, z.imag,label="Deux premiers termes (k = {}, h = {})".format(k,h), color="red")
plt.plot(z1.real, z1.imag, '--', label="Tous les termes (k = {}, h = {})".format(k,h), color="red")



''' plot cercle unité complexe '''
t = np.linspace(0,2*np.pi,101)
plt.plot(np.cos(t),np.sin(t),label="Cercle unité", color="black")

''' axes horizontal et vertical '''
m = max(max(abs(z1.real)),max(abs(z1.imag)))
plt.plot([0,0],[-1.1*m,1.1*m],'k',linewidth=0.5)
plt.plot([-1.1*m,1.1*m],[0,0],'k',linewidth=0.5)
plt.xlabel("Partie réelle")
plt.ylabel("Partie imaginaire")
plt.legend()
plt.axis('equal')
plt.xlim(-1.1*m,1.1*m),plt.ylim(-1.1*m,1.1*m)
plt.title("Facteur d'amplification $\kappa$ dans le cercle unité")
plt.show()




#################### kappa^2 <= 1 en fonction du cos(rh) ####################
r = np.linspace(0,2*np.pi,101)
h2 = 1

############### table 1
plt.subplot(221)
alpha = -1.5
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
#z2 = 1 - ((1-np.cos(r*h2))*(2*alpha-2*np.cos(r*h2)*alpha**2)) - (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
#z = 1 - ((1-np.cos(r*h))*(2*alpha-2*alpha**2)) + (np.sin(r*h)**2)*((np.cos(r*h)-1)*(16*beta*beta*(np.cos(r*h)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))
#plt.plot(np.cos(r*h2), z2,label="2: alpha = {}, beta = {}".format(alpha,beta))

alpha = -1.2
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = -1.0
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = -1.0
beta = 0.07
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = -0.5
beta = 0.05
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = -0.1
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

plt.xlim(-1,1)
plt.xlabel("$cos(rh)$")
plt.ylabel("$|\kappa |^2$")
plt.legend()

############### table 2
plt.subplot(222)
alpha = 0.0
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
#z = 1 - ((1-np.cos(r*h))*(2*alpha-2*alpha**2)) + (np.sin(r*h)**2)*((np.cos(r*h)-1)*(16*beta*beta*(np.cos(r*h)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))


alpha = 0.05
beta = 0.05
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.1
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.2
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.5
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.5
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

plt.xlim(-1,1)
plt.xlabel("$cos(rh)$")
plt.ylabel("$|\kappa |^2$")
plt.legend()

############### table 3
plt.subplot(223)
alpha = 0.6
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
#z = 1 - ((1-np.cos(r*h))*(2*alpha-2*alpha**2)) + (np.sin(r*h)**2)*((np.cos(r*h)-1)*(16*beta*beta*(np.cos(r*h)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.8
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.9
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 0.9
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.0
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.0
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

plt.xlim(-1,1)
plt.xlabel("$cos(rh)$")
plt.ylabel("$|\kappa |^2$")
plt.legend()


############### table 4
plt.subplot(224)
alpha = 1.05
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
#z = 1 - ((1-np.cos(r*h))*(2*alpha-2*alpha**2)) + (np.sin(r*h)**2)*((np.cos(r*h)-1)*(16*beta*beta*(np.cos(r*h)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.1
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.1
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.2
beta = 0.1
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.5
beta = 0.05
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

alpha = 1.8
beta = 0.0
z = 1 - ((1-np.cos(r*h2))*(2*alpha-2*alpha**2)) + (np.sin(r*h2)**2)*((np.cos(r*h2)-1)*(16*beta*beta*(np.cos(r*h2)-1)+8*alpha*beta))
plt.plot(np.cos(r*h2), z,label="alpha = {}, beta = {}".format(alpha,beta))

plt.xlim(-1,1)
plt.xlabel("$cos(rh)$")
plt.ylabel("$|\kappa |^2$")
plt.legend()

plt.suptitle("Facteur d'amplification $\kappa ^2$ en fonction du terme $cos(rh)$ pour le critère de stabilité")
plt.show()

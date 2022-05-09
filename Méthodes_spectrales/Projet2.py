""" Import """
import numpy as np
import matplotlib.pyplot as plt


def compute(r, L):

    N = 1024
    x = np.linspace(0, L - L / N, N)
    dt = 0.05
    Nt = int(200 / dt)
    t = np.linspace(0, 200, Nt)

    un0 = np.cos(2 * np.pi * x / L) + 0.1 * np.cos(4 * np.pi * x / L)
    uk0 = np.fft.fftshift(np.fft.fft(un0))

    k = np.linspace(-N / 2, N / 2 - 1, N)

    uk = np.zeros((Nt, N), dtype=complex)
    un = np.zeros((Nt, N))
    uk[0] = uk0
    un[0] = un0

    fl = r - 1 + 2 * ((2 * np.pi * k / L)**2) - ((2 * np.pi * k / L)**4)

    #Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[0]))).real)**3))
    Fu = np.fft.fftshift(np.fft.fft(un[0]**3))
    uk[1] = ((1 + 0.5 * dt * fl) / (1 - 0.5 * dt * fl)) * \
        uk[0] + dt * ((Fu) / (1 - 0.5 * dt * fl))
    un[1] = (np.fft.ifft(np.fft.ifftshift(uk[1]))).real

    for i in range(2, Nt):
        #Fu = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-1]))).real)**3))
        #FuAv = np.fft.fftshift(np.fft.fft(((np.fft.ifft(np.fft.ifftshift(uk[i-2]))).real)**3))
        Fu = np.fft.fftshift(np.fft.fft(un[i - 1]**3))
        FuAv = np.fft.fftshift(np.fft.fft(un[i - 2]**3))
        uk[i] = ((1 + 0.5 * dt * fl) / (1 - 0.5 * dt * fl)) * uk[i -
                                                                 1] - dt * ((1.5 * Fu - 0.5 * FuAv) / (1 - 0.5 * dt * fl))
        un[i] = (np.fft.ifft(np.fft.ifftshift(uk[i]))).real
    return un


r = 0.2
L = 200

N = 1024
x = np.linspace(0, L - L / N, N)
dt = 0.05
Nt = int(200 / dt)
t = np.linspace(0, 200, Nt)

un = compute(r, L)

c = np.linspace(np.min(un[-1]), np.max(un[-1]), 101)
c = np.linspace(-1, 1, 101)
[xx, tt] = np.meshgrid(x, t)
plt.contourf(xx, tt, un, c, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(0, 175)
plt.title("r = {}, L = {}".format(r, L))
plt.show()

# ---------------- Plot dépendance ----------------

# # Dépendance en r

# variation amplitude
amplitudes = []
lengths = []
# rs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# Ls = [100,125,150,175, 200]

# for k in range(len(rs)) :
#     U = compute(rs[k], L)
#     amplitudes.append(np.max(U[125]))

# for k in range(len(Ls)) :
#     U = compute(0.2,Ls[k])
#     lengths.append(np.max(U[200]))


# plt.plot(rs,amplitudes,'-o')
# plt.title("Variation de l'amplitude du champ de température en fonction de r")
# plt.xlabel('r')
# plt.ylabel('Amplitude de u(x,t)')
# plt.show()


# plt.plot(Ls,lengths,'-o')
# plt.title("Variation de l'amplitude du champ de température en fonction de L")
# plt.xlabel('L')
# plt.ylabel('Amplitude de u(x,t)')
# plt.ylim(-0.01,0.08)
# plt.show()

# Ls = [10, 50, 100, 125, 150, 175, 200]
# rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# lambd = []


# oldMax = 0
# # -------------- /!\ ATTENTION !! UTILISER np.round(numer, numberof_decimals) POUR MATCHER LE MAX
# for k in range(len(Ls)):
#     count = 0
#     Ul = compute(0.2, Ls[k])
#     x = np.linspace(0, Ls[k] - Ls[k] / N, N)
#     lim = np.round(np.max(Ul[-1]), 2)
#     print("max = ", lim)
#     for j in range(len(Ul[-1])):
#         # print(Ul[-1][j])
#         if((np.round(Ul[-1][j], 2) == lim) & count == 1):
#             print("here")
#             print("x[j] = ", x[j])
#             lambd.append(x[j] - oldMax)
#             break
#         if((np.round(Ul[-1][j], 2) == lim) & count == 0):
#             print("oldMax = ", oldMax)
#             oldMax = x[j]
#             count += 1


# # Ur = compute(rs[k], 100)
# # lengths.append(np.max(Ul[200]))
# # amplitudes.append(np.max(Ur[125]))
# print(lambd)
# print(Ls)

# plt.plot(Ls, lambd, '-o')
# plt.title("Variation de la longueur d'onde en fonction de L")
# plt.xlabel('L')
# plt.ylabel('$\lambda$')
# plt.show()

# print(lambd)
# print(rs)

# plt.plot(rs, lambd, '-o')
# plt.title("Variation de la longueur d'onde en fonction de r")
# plt.xlabel('r')
# plt.ylim(-0.2, 0.2)
# plt.ylabel('$\lambda$')
# plt.show()

# plt.plot(Ls,lengths,'-o')
# plt.title("Variation de l'amplitude du champ de température en fonction de L")
# plt.xlabel('L')
# plt.ylabel('Amplitude de u(x,t)')
# plt.ylim(-0.01,0.08)
# plt.show()

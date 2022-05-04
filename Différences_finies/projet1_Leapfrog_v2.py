''' Import Module'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display


dx = 0.1739
N = int(40 / dx) + 1
x = np.arange(-20, 20, dx)
dt = 0.002
Nt = int(70 / dt)
t = np.arange(0, 70, dt)
delta = 0.4

A = 1
k = (A / 2)**0.5
w = 4 * k**3

un = np.zeros((Nt, N))
un0 = A / (np.cosh(k * x)**2)
un[0] = un0

# leap-frog (Zabusky et Kruskal)
un[1] = un0
for i in range(0, N - 2):
    un[1][i] = un[0][i] - (dt / dx) * (un[0][i + 1] + un[0][i] + un[0][i - 1]) * (un[0][i + 1] - un[0]
                                                                                  [i - 1]) - (dt / (2 * dx**3)) * (un[0][i + 2] - 2 * un[0][i + 1] + 2 * un[0][i - 1] - un[0][i - 2])
un[1][N - 2] = un[0][N - 2] - (dt / dx) * (un[0][N - 1] + un[0][N - 2] + un[0][N - 3]) * (un[0][N - 1] -
                                                                                          un[0][N - 3]) - ((dt) / (2 * dx**3)) * (un[0][0] - 2 * un[0][N - 1] + 2 * un[0][N - 3] - un[0][N - 4])
un[1][N - 1] = un[0][N - 1] - (dt / dx) * (un[0][0] + un[0][N - 1] + un[0][N - 2]) * (
    un[0][0] - un[0][N - 2]) - ((dt) / (2 * dx**3)) * (un[0][1] - 2 * un[0][0] + 2 * un[0][N - 2] - un[0][N - 3])

for j in range(1, Nt - 1):
    for i in range(0, N - 2):
        un[j + 1][i] = un[j - 1][i] - (dt / (3 * dx)) * (un[j][i + 1] + un[j][i] + un[j][i - 1]) * (un[j][i + 1] - un[j][i - 1]) - (
            (dt * delta**2) / (dx**3)) * (un[j][i + 2] - 2 * un[j][i + 1] + 2 * un[j][i - 1] - un[j][i - 2])
    un[j + 1][N - 2] = un[j - 1][N - 2] - (dt / (3 * dx)) * (un[j][N - 1] + un[j][N - 2] + un[j][N - 3]) * (
        un[j][N - 1] - un[j][N - 3]) - ((dt * delta**2) / (dx**3)) * (un[j][0] - 2 * un[j][N - 1] + 2 * un[j][N - 3] - un[j][N - 4])
    un[j + 1][N - 1] = un[j - 1][N - 1] - (dt / (3 * dx)) * (un[j][0] + un[j][N - 1] + un[j][N - 2]) * (
        un[j][0] - un[j][N - 2]) - ((dt * delta**2) / (dx**3)) * (un[j][1] - 2 * un[j][0] + 2 * un[j][N - 2] - un[j][N - 3])


time = 0
plt.plot(x, un[time], label='t = 0')

time = int(Nt * 0.25 / (1))
plt.plot(x, un[time], label='t = 0.25')

time = int(Nt * 0.5 / (1))
plt.plot(x, un[time], label='t = 0.5')

time = int(Nt / (1) - 1)
plt.plot(x, un[time], label='t = 1')
plt.legend()
plt.show()


for n in range(0, Nt, int(Nt / 25)):
    if n == 0:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    plt.clf()
    plt.plot(x, un[n, :])
    plt.gca()
    plt.title('Simulation')
    plt.xlabel('x')
    plt.ylabel('u')
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

# # initializing a figure
# fig = plt.figure()

# # labeling the x-axis and y-axis
# axis = plt.axes(xlim=(0, 4),  ylim=(-1.5, 1.5))

# # initializing a line variable
# line, = axis.plot([], [], lw=3)

# def animate(frame_number):
#     x = np.linspace(0, 4, 1000)

#     # plots a sine graph
#     y = np.sin(2 * np.pi * (x - 0.01 * frame_number))
#     line.set_data(x, y)
#     line.set_color('green')
#     return line,


# anim = animation.FuncAnimation(fig, animate, frames=100,
#                                interval=20, blit=True)
# fig.suptitle('Sine wave plot', fontsize=14)

# # converting to an html5 video
# video = anim.to_html5_video()

# # embedding for the video
# html = display.HTML(video)

# # draw the animation
# display.display(html)
# plt.close()

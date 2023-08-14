import os
import numpy as np
import matplotlib.pyplot as plt

# create map layout
map_grid = np.ones((100,100))
map_grid[20:80, 20:40] = 0 # shelf 1
map_grid[20:80, 60:80] = 0 # shelf 2

plt.imshow(map_grid, cmap='gray')
plt.title('Grocery Map')

drag_start = []
def onpress(event):
    drag_start.append((int(event.xdata), int(event.ydata)))

drag_end = []
def onrelease(event):
    drag_end.append((int(event.xdata), int(event.ydata)))

cidpress = plt.gcf().canvas.mpl_connect('button_press_event', onpress)
cidrelease = plt.gcf().canvas.mpl_connect('button_release_event', onrelease)
plt.show()

xs = []
ys = []
for s, e in zip(drag_start, drag_end):
    minx = min(s[0], e[0]) 
    maxx = max(s[0], e[0])
    miny = min(s[1], e[1])
    maxy = max(s[1], e[1])
    xs.append((minx, maxx))
    ys.append((miny, maxy))



fig, ax = plt.subplots()
ax.imshow(map_grid, cmap='gray')
for (minx, maxx), (miny, maxy) in zip(xs, ys):
    rect = plt.Rectangle((minx, miny), maxx - minx, maxy - miny, 
                        facecolor='none', edgecolor='red')
    ax.add_patch(rect)
plt.show()
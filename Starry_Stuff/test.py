import starry
import numpy as np

starry.config.lazy = False

map = starry.Map(ydeg=5)

map[5, -3] =1

print(map.y)

theta = np.linspace(0, 360, 50)
map.show(theta=theta)

map[5, -3] = -1

print(map.y)

theta = np.linspace(0, 360, 50)
map.show(theta=theta)


map[5, -3] = 0

print(map.y)

theta = np.linspace(0, 360, 50)
map.show(theta=theta)


import matplotlib.pyplot as plt
import numpy as np

xpoint = np.array([0,1,2,3,4])
ypoint = np.array([0,1,4,9,16])
plt.plot(xpoint,ypoint,ls = 'dashed') # ls - linestyle
plt.title("X-Y coordinate system",loc="right") # 图的title,loc-location 的参数
plt.xlabel("x-cm") # x的label
plt.ylabel("y-cm") # y的label
plt.grid() # 图的网格
plt.show()

def function(x):
    return x^2

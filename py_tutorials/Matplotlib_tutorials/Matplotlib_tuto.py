import matplotlib.pyplot as plt
import numpy as np

xpoint = np.array([0,1,2,3,4])
ypoint = np.array([0,1,4,9,16])
plt.plot(xpoint,ypoint,scalex=20,scaley=20)
plt.title = "X-Y coordinate system"

plt.show()

def function(x):
    return x^2

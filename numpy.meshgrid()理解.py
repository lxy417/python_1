import numpy as np
import matplotlib.pyplot as plt
x=np.array([[0,1,2],[0,1,2]])
y=np.array([[0,0,0],[1,1,1]])

plt.plot(x,y,color='red',marker='.',linestyle=' ')
plt.grid(True)
plt.show()

x1=np.array([0,1,2])
y1=np.array([0,1])
X,Y=np.meshgrid(x1,y1)
print(X)
print(Y)

x2=np.linspace(0,1000,20)
y2=np.linspace(0,500,20)
X,Y=np.meshgrid(x2,y2)
plt.plot(X,Y,
         color='red', marker='.', linestyle=' ')
plt.show()
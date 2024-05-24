import matplotlib.pyplot as plt
import numpy as np

# Read data from file
data = np.loadtxt('output.txt')

# Create 2D grid
X = data[:, 0].reshape(-1, 81)
Y = data[:, 1].reshape(-1, 81)
Z = data[:, 2].reshape(-1, 81)

# Create a contour plot
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.title('Output of finalWaveFunction.feedForward()')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

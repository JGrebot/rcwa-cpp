import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: %s <filename>", sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]

matrix = np.loadtxt(filename)
plt.matshow(matrix)
plt.colorbar()
plt.show()

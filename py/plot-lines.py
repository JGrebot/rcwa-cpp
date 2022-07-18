import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: %s <filename>", sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]

f = open(filename, "r")

x = [float(s) for s in f.readline().strip().split(" ")]
ys = []

while True:
    line = f.readline().strip()
    if len(line) == 0: break
    line_data = [float(s) for s in line.split(" ")]
    ys.append(line_data)

for y in ys:
    plt.plot(x, y)
plt.show()


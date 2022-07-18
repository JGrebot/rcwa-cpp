import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: %s <filename> <field-index>" % sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]
field_index = int(sys.argv[2])

fields_names = {
    "PLANAR_1D_TM": ["Hy", "Ex", "Ez"],
    "PLANAR_1D_TE": ["Ey", "Hx", "Hz"],
}

f = open(filename, "r")
mode = f.readline().rstrip()
if mode not in fields_names:
    print("Error: unknown mode: %s" % mode)
if field_index < 0 or field_index > 2:
    print("Error: invalid field index: %d. Should be 0, 1 or 2." % field_index)
print("%s: printing %s field" % (mode, fields_names[mode][field_index]))

x = [float(s) for s in f.readline().strip().split(" ")]
z = [float(s) for s in f.readline().strip().split(" ")]

field = ([], [], [])
i = 0
for line in f:
    row_values = [float(s) for s in line.strip().split(" ")]
    if len(row_values) > 0:
        field[i % 3].append(row_values)
    i += 1
matrix = np.array(field[field_index])
plt.pcolormesh(x, z, matrix)
plt.colorbar()
plt.title("Real part of %s" % fields_names[mode][field_index])
plt.show()

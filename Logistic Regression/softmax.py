import numpy as np

z = np.array([5, 3, 0, -1])
n = z.shape[0]

ez = np.exp(z)
print(ez)
sum_ez = ez.sum()
print(sum_ez)
result = ez / sum_ez
print(result)

for row in range(n):
    for col in range(n):
        if row == col:
            v = result[row] * (1.0 -  result[row])
        else:
            v = -1.0 * result[row] * result[col]
        print(f"{v:.7f} & ", end="")
    print(f"\\\\")


import numpy as np

# Fixed Input (8x8)
inp = np.array([
[1,0,1,1,0,0,1,0],
[0,1,0,1,1,0,0,1],
[1,1,0,0,1,1,0,0],
[0,0,1,1,0,1,1,0],
[1,0,0,1,1,0,1,1],
[0,1,1,0,0,1,0,1],
[1,1,0,1,0,0,1,0],
[0,0,1,0,1,1,0,1]
])

# Fixed Kernel (3x3)
kernel = np.array([
[1,0,1],
[0,1,0],
[1,0,1]
])

def xnor(a, b):
    return ~(a ^ b) & 1

def binary_conv2d(input, kernel):
    H, W = input.shape
    kH, kW = kernel.shape
    out = np.zeros((H - kH + 1, W - kW + 1), dtype=int)

    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            window = input[i:i+kH, j:j+kW]
            xnor_res = xnor(window, kernel)
            out[i, j] = np.sum(xnor_res)

    return out

output = binary_conv2d(inp, kernel)

print("Input:\n", inp)
print("Kernel:\n", kernel)
print("Output:\n", output)

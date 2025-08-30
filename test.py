import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import time

N = 2048

matrix_a = np.random.rand(N, N)
matrix_b = np.random.rand(N, N)

start = time.time()
result_matrix = np.matmul(matrix_a, matrix_b)
end = time.time()
diff = end-start

print(diff)

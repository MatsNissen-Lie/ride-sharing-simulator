import numpy as np
import timeit

N = 100000
# 1. Tuple Dict
d_tuple = {(i, j): 1.0 for i in range(N) for j in range(N)}
# 2. Nested Dict
d_nested = {i: {j: 1.0 for j in range(N)} for i in range(N)}
# 3. Flat List
flat_list = [1.0] * (N * N)
# 4. NumPy
arr = np.zeros((N, N))

# Accessing index (500, 500)
t_tuple = timeit.timeit("d[(500, 500)]", globals={"d": d_tuple}, number=1000000)
t_nested = timeit.timeit("d[500][500]", globals={"d": d_nested}, number=1000000)
t_list = timeit.timeit("l[500 * 1000 + 500]", globals={"l": flat_list}, number=1000000)
t_numpy = timeit.timeit("a[500, 500]", globals={"a": arr}, number=1000000)

print(f"Flat List:   {t_list:.4f} s (Winner)")
print(f"Tuple Dict:  {t_tuple:.4f} s")
print(f"Nested Dict: {t_nested:.4f} s")
print(f"NumPy:       {t_numpy:.4f} s (Slowest for single read)")

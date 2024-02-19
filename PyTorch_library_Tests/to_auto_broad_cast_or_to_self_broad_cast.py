"""To auto broad cast or to self broad cast? That is the question.

This script compares the time for matrix multiplication with automatic broadcasting and einsum.
It also compares the time for matrix multiplication on CPU and GPU. The argument being that self broadcasting and einsum are 
faster than the automatic broadcasting and matmul functions in torch.

Results: For large matrices on the GPU, self broadcasting was faster for me by a difference of 4 seconds faster for 
a matrix of 21.5 GB - For CPU it seems auto broad caster was faster.

TLDR: For large matrices on the GPU, self broadcasting is faster than auto broad casting. Manage your shapes yourself!
"""

import torch
import time

# Initialize tensors
t1 = torch.rand(1750, 1750)
t2 = torch.rand(1750, 1750, 1750)
# Print the memory size of t2 in GB
print(f"Memory size of t2: {t2.element_size() * t2.nelement() / 1e9:.4f} GB")

# Measure time for matmul with automatic broadcasting
start_time_auto = time.time()
t3_auto = torch.matmul(t1, t2)
time_auto = time.time() - start_time_auto

# Measure time for matmul with einsum
start_time_einsum = time.time()
t3_einsum = torch.einsum('ij,jkl->ikl', t1, t2)
time_einsum = time.time() - start_time_einsum

print(f"Time for matmul with automatic broadcasting: {time_auto:.4f} seconds")
print(f"Time for matmul with einsum: {time_einsum:.4f} seconds")

# Move to GPU and compare time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t1 = t1.to(device)
t2 = t2.to(device)

# Measure time for optimized matmul with einsum
start_time_einsum_opt = time.time()
t3_einsum_opt = torch.einsum('ij,jkl->ikl', t1, t2)
time_einsum_opt = time.time() - start_time_einsum_opt

# Meaure time for GPU matmul with automatic broadcasting
start_time_auto_gpu = time.time()
t3_auto_gpu = torch.matmul(t1, t2)
time_auto_gpu = time.time() - start_time_auto_gpu

# print gpu times
print(f"Time for matmul with optimized einsum on gpu: {time_einsum_opt:.4f} seconds")
print(f"Time for matmul with automatic broadcasting on gpu: {time_auto_gpu:.4f} seconds")

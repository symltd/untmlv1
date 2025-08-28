The UltraEfficientSparseFFN is conceptually sparse, but our FLOPs hooks and PyTorch execution still compute all matrix operations in the forward pass. So the actual speedup from sparsity is not fully realized on GPU unless:

- We replace nn.Linear operations with true sparse kernels.
- We optimize FFT, polynomial, and micro-refinement operations to skip zeroed elements in hardware-efficient ways.

Sparse GPT-2 increases steps because the batch may be smaller for memory constraints (hidden size larger or multiple sparse transformations), which increases total runtime.

#### Next Steps for Accurate Benchmarking
1. Hook into actual kernel operations (e.g., count only nonzero multiplications in spectral/polynomial/micro layers).
2. Profile GPU kernel execution (e.g., torch.cuda.Event, torch.profiler) rather than just FLOPs.
3. Ensure batch sizes and data splits are identical between dense and sparse runs for fair timing comparison.
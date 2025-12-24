# Faster Vertex Cover Algorithms on GPU
Faster vertex cover algorithms on GPUs with component-aware parallel branching by Hussein Amro, Basel Fakhri, Amer E. Mouawad, and Izzat El Hajj.

# Reference
- H. Amro, B. Fakhri, A. E. Mouawad and I. E. Hajj, "Faster Vertex Cover Algorithms on GPUs With Component-Aware Parallel Branching," in IEEE Transactions on Parallel and Distributed Systems, vol. 37, no. 2, pp. 504-517, Feb. 2026, doi: [10.1109/TPDS.2025.3641049](https://doi.org/10.1109/TPDS.2025.3641049).

Please cite this paper if you find our work useful.

# Description
CUDA/C++ implementation of a component-aware, load-balanced GPU solver for Minimum Vertex Cover and its parameterized (k-VC) variant. It detects connected-component splits on the fly, parallelizes non-tail-recursive branches via last-descendant aggregation, and applies graph reductions to lower memory use—achieving seconds-scale runtimes where prior GPU methods exceed hours [1].

# Features
1- Load balancing via a global worklist `GlobalWorkList` to keep all blocks busy. <br>
2- Component-aware branching that detects connected components on the fly and solves them independently to avoid redundant work, managed by the Component Branch Registry `compArray`. <br>
3- Occupancy-aware launch that considers input size and device limits to maximize GPU utilization and select an optimal configuration. <br>
4- Preprocessing on the CPU, including crown reduction, to form an induced subgraph that the GPU can process with lower memory and higher parallelism. <br>
5- Placing the per-node degree array in shared memory when it fits at runtime and falling back to global memory otherwise, to reduce access latency. <br>
6- Selecting `int16_t` over `int32_t` for the degree array entries when the runtime max degree permits, shrinking memory per block, and enabling more concurrent workers. <br>

# Structure

**Core Modules**
- **main.cu** Orchestrates execution: parses config for CSR graph construction, applies heuristics & reductions, then selects the sequential or GPU path and launches the appropriate kernels.
- **GlobalWorkList.cuh**
  Implements the global-worklist exact MVC kernels, providing dynamic load balancing across blocks/SMs during branching.
- **GlobalWorkListParameterized.cuh**
  Implements the global-worklist kernels for the parameterized `k`-Vertex Cover (PVC), with dynamic load balancing and `k`-aware pruning.
- **LocalStacks.cuh**
  Implements the stack-only exact MVC kernels—per-block local stacks with no global worklist and no dynamic load balancing.
- **LocalStacksParameterized.cuh**
  Implements stack-only kernels for parameterized `k`-Vertex Cover, using per-block local stacks and `k`-aware pruning with no global worklist and no dynamic load balancing.
- **Sequential.cpp**
  Implements the CPU baseline: a sequential exact Minimum Vertex Cover solver (branch-and-reduce/DFS) for comparison with GPU kernels.
- **SequentialParameterized.cpp**
  Implements the CPU baseline for parameterized `k`-Vertex Cover using branch-and-reduce with `k`-aware pruning.

**Data Structures**
- **CSRGraphRep.cuh** Defines the CSR graph structure and device functions to allocate, copy, and free GPU-side arrays (`dst`, `srcPtr`, `degree`).
- **BWDWorkList.cuh**
Implements a ticketed ring-buffer global worklist for dynamic inter-block load balancing—atomic head/tail, per-slot `tickets`, combined counters (`numWaiting`/`numEnqueued`), exponential backoff (`__nanosleep`), thresholds, and PVC/connected-components enqueue/dequeue variants, adapted from [2].
- **stack.cuh**
Defines per-block GPU stacks with device `push`/`pop` (and connected-components variant) plus alloc/free, storing vertex-degree snapshots, deleted-counts, array indices, and flags.
- **compArray.cuh**
Implements Component Branch Registry for Component-Aware Parallel Branching.

**Reduction Algorithms**
- **helperFunctions.cpp** Implements sequential reduction rules (`leaf`, `highDegree`, `triangle`, `clique`, `cycle`), `deleteVertex`, and utilities used by the CPU MVC solver.
- **helperFunctions.cuh**
Implements device functions reduction rules, max-degree branching, and component discovery (`BFS`/`SCC`) for stack-only and globalWorkList GPU kernels.
- **crown.cpp**
Implements the root-node crown reduction to shrink the graph before launching the sequential solvers or GPU kernels.
- **hopcroft_karp.cpp**
Implements the Hopcroft–Karp algorithm for maximum bipartite matching, powering the crown reduction’s matching step.

**Utilities**
- **Counters.cuh** Per-block profiling: enum-based per-block counters using `clock64` to time each algorithm step, max-depth tracking, and host-side CSV dumps plus per-SM node counts.
- **auxFunctions.cpp**
Pre-kernel setup: builds the CSR graph, approximates MVC heuristics, then selects blocks/threads and shared vs global memory via `setBlockDimAndUseGlobalMemory`.

## Experimental Setup
 - We implement our code using C++ and CUDA, and compile it with nvcc from the CUDA SDK version 11.7. We evaluate our CPU implementation using an AMD EPYC 7551P CPU with 128GB of main memory. We evaluate our GPU implementations using a Volta V100 GPU with 32GB of device memory.
 - The codebase is designed to be portable across NVIDIA GPUs, automatically adapting to the available hardware to maximize resource utilization and performance.

# Folders
- data: Contains the graph datasets used for evaluation [3],[4].
- src: Contains the source code of the project.

# Instructions
All of the commands below should be executed in the src folder

- To compile:
```make```

- To compile enabling Counters (won't run efficiently):
```make USE_COUNTERS=1```

- To clean:
```make clean```

- To run:
```./output <args>```

- For help on how to configure arguments:
```./output -h ```

# Notes
- Data regarding your run will be appended as a row to the end of the a csv file Results.csv in src/Results/Results.csv
- If code is compiled with enabling counters, then counter data about each block will be written in files in src/NODES_PER_SM and src/Counters for each graph.
- Data in the src/NODES_PER_SM folder represents how many nodes from the work tree each SM in your GPU solved.
- Data in the src/Counters folder represents how much time was collectively spent on parts of the code.

## References

<a id="1">[1]</a> P. Yamout, K. Barada, A. Jaljuli, A. Mouawad, I. El Hajj. Parallel Vertex Cover Algorithms on GPUs. In Proceedings of the IEEE International Parallel & Distributed Processing Symposium (IPDPS), 2022. [DOI](https://arxiv.org/pdf/2204.10402)

<a id="2">[2]</a> B. Kerbl et al., “The broker queue: A fast, linearizable FIFO queue for fine-granular work distribution on the GPU,” in Proceedings of the 2018 International Conference on Supercomputing, 2018, pp. 76–85. [DOI](https://arbook.icg.tugraz.at/schmalstieg/Schmalstieg_353.pdf)

<a id="3">[3]</a>  R. A. Rossi and N. K. Ahmed, “The network data repository with interactive graph analytics and visualization,” in AAAI, 2015. [Online]. Available: https://networkrepository.com

<a id="4">[4]</a> M. A. Dzulfikar et al., “The pace 2019 parameterized algorithms and computational experiments challenge: the fourth iteration,” in 14th International Symposium on Parameterized and Exact Computation (IPEC 2019). Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.

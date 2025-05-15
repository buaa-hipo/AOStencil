# AOStencil - ARM-Optimized Stencil Computation Compiler

A high-performance DSL compiler for stencil computations optimized for ARM architectures. The source code of [Efficient Locality-aware Instruction Stream Scheduling for Stencil Computation on ARM Processors](10.1145/3721145.3725760)

## Features
- 2D/3D stencil computation optimization
- Locality memory allocation and thread binding
- With independent instruction stream scheduling with the Serial-FMA to Tree-Based Reduction (SFTBR) method.
- Automatic kernel tuning using genetic algorithms
- Multi-threaded execution with pthreads
- ARM NEON SIMD vectorization support
- Domain Specific Language (DSL) for stencil definition
- Cross-platform support (tested on Phytium, Kunpeng ARM processors)
- **More details are available in ics25 paper.**

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install gcc libnuma-dev
```

# Python package

## Installation

```bash
cd AOStencil
pip install -e .
```
## Usage

### Basic Usage
```python
from aostencil import Stencil2d, kernel_tune_stencil_2d, gen_stencil_2d
import numpy as np

# Create coefficient matrix for 9-point stencil
coefficients = np.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1],
    [0.1, 0.1, 0.1]
])
# star shape like coefficients = np.array([
# [0,0,.1,0,0],
# [0,0,.2,0,0],
# [.1,.2,.4,.2,.1],
# [0,0,.2,0,0],
# [0,0,.1,0,0]])

# Initialize 2D stencil with parameters:
# col=8194, row=8192 (grid dimensions)
# coefficients, edge_length=1 (from 3x3 kernel)
# datatype='float' (single precision)
stencil = Stencil2dIR(
    8192,8194,coefficients,0,'float'
)
stencil.set_name("2d9pt_box")

# Configure NUMA topology (16 nodes, 8 cores each)
stencil.set_numa_config(16, 8)
stencil.set_run_config(16, 8)

# Auto-tune for optimal parameters
optimized_stencil = kernel_tune_stencil_2d(stencil)

# Generate optimized C code
kernel_code = gen_stencil_2d(optimized_stencil)

# Save generated kernel
with open('optimized_kernel.c', 'w') as f:
    f.write(kernel_code)
```
**You can find more examples in benchmark directory.**


### DSL 
```python
from aostencil import from_dsl_load_stencil, kernel_tune_stencil_2d

# Load stencil from DSL
with open("stencil.dsl") as f:
    stencil = from_dsl_load_stencil(f.read())
    
# Configure hardware topology (NUMA nodes x cores per node)
stencil.set_numa_config(16, 8)  # Example for Phytium FT-2000+
stencil.set_run_config(16, 8)   # 16 NUMA nodes, 8 cores per node

# Auto-tune and generate optimized kernel
optimized_stencil = kernel_tune_stencil_2d(stencil)
```

### Fine-Tuning
For advanced optimization control, directly use the tuning classes:

```python
from aostencil import Stencil2dIR, stencil2d_opt_search

# Create search instance with custom parameters
tuner = stencil2d_opt_search(
    stencil2d=Stencil2dIR(...),      # Your stencil configuration
    cache_path='tuning_cache',    # Cache for generated kernels
    population_size=200,         # Larger population
    mutation_rate=0.2,            # Balance exploration/exploitation
    test_time_per_iter=50         # More iterations 
)

# Run optimization (returns best found configuration)
optimized_stencil, exec_time = tuner.search(record_log=True)
```
## Benchmark
The benchmark directory contains pre-configured stencil implementations for different ARM architectures:
```bash
benchmark/
├── dsl/                   # DSL definition examples
│   ├── 2d9pt_box.dsl      # 2D 9-point box stencil DSL  
│   ├── 3d7pt_star.dsl     # 3D 7-point star stencil DSL
│   └── dsl_example.py     # DSL loading examples
│
├── phytium/               # Phytium FT-2000+ optimized benchmarks
│   ├── 2d_*.py             # 2D stencils
│   ├── 3d_*.py             # 3D stencils
│   ├── run.sh             # Batch execution script
│   └── str_main.py        # Verification kernels
│
├── kunpeng/               # Kunpeng 920 optimized benchmarks
│   ├── 2d_*.py             
│   ├── 3d_*.py
│   └── ...                # Same structure as phytium
│
├── phytium_f64/                   # phytium float64 benchmark
└── kunpeng_f64/                   # kunpeng float64 benchmark                 
```

## License
This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.


## Citation
If you use this project in your research, please cite our paper:
```bibtex
@inproceedings{liu2025aostencil,
  title={Towards Efficient Instruction Stream Scheduling for Stencil Computation on ARM Processors},
  author={Shanghao Liu, Hailong Yang, Xin You, Zhongzhi Luan, Yi Liu, Depei Qian},
  booktitle={2025 International Conference on Supercomputing (ICS '25), June 8--11, 2025, Salt Lake City, UT, USA},
  year={2025}
}
```
@info "Testing CUDA"
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.versioninfo()

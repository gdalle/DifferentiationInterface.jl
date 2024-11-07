@info "Trying out CUDA"
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.versioninfo()

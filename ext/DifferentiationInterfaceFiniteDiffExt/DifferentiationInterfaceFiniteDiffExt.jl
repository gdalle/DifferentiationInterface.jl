module DifferentiationInterfaceFiniteDiffExt

using ADTypes: AutoFiniteDiff, AutoSparseFiniteDiff
import DifferentiationInterface as DI
using FiniteDiff:
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_hessian,
    finite_difference_hessian!,
    finite_difference_jacobian,
    finite_difference_jacobian!
using LinearAlgebra: dot, mul!

const AllAutoFiniteDiff = Union{AutoFiniteDiff,AutoSparseFiniteDiff}

# see https://github.com/SciML/ADTypes.jl/issues/33

fdtype(::AutoFiniteDiff{fdt}) where {fdt} = fdt
fdjtype(::AutoFiniteDiff{fdt,fdjt}) where {fdt,fdjt} = fdjt
fdhtype(::AutoFiniteDiff{fdt,fdjt,fdht}) where {fdt,fdjt,fdht} = fdht

fdtype(::AutoSparseFiniteDiff) = Val{:central}()
fdjtype(::AutoSparseFiniteDiff) = Val{:central}()
fdhtype(::AutoSparseFiniteDiff) = Val{:hcentral}()

# see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions
const FUNCTION_INPLACE = Val{true}
const FUNCTION_NOT_INPLACE = Val{false}

include("allocating.jl")
include("mutating.jl")

end # module

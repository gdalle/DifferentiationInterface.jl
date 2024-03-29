module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
import DifferentiationInterface as DI
using ForwardDiff.DiffResults: DiffResults, DiffResult, GradientResult
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
    GradientConfig,
    HessianConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: dot, mul!

include("utils.jl")
include("allocating.jl")
include("mutating.jl")

end # module

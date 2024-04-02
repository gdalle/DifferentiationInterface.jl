module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff, AutoSparseForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras, GradientExtras, HessianExtras, JacobianExtras, NoPushforwardExtras
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

const AnyAutoForwardDiff{C,T} = Union{AutoForwardDiff{C,T},AutoSparseForwardDiff{C,T}}

include("utils.jl")
include("allocating.jl")
include("mutating.jl")

end # module

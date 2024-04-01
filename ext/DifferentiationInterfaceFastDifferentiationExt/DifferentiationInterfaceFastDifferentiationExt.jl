module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoFastDifferentiation, AutoSparseFastDifferentiation
import DifferentiationInterface as DI
using FastDifferentiation:
    derivative,
    hessian,
    hessian_times_v,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables,
    sparse_hessian,
    sparse_jacobian
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

const AnyAutoFastDifferentiation = Union{
    AutoFastDifferentiation,AutoSparseFastDifferentiation
}

DI.mode(::AnyAutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.supports_mutation(::AnyAutoFastDifferentiation) = DI.MutationNotSupported()

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

issparse(::AutoFastDifferentiation) = false
issparse(::AutoSparseFastDifferentiation) = true

include("allocating.jl")

end

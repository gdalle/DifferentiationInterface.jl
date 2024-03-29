module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoFastDifferentiation, AutoSparseFastDifferentiation
import DifferentiationInterface as DI
using FastDifferentiation:
    derivative,
    hessian,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

const AnyAutoFastDifferentiation = Union{
    AutoFastDifferentiation,AutoSparseFastDifferentiation
}

DI.mode(::AnyAutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.supports_mutation(::AnyAutoFastDifferentiation) = DI.MutationNotSupported()

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

include("allocating.jl")

end

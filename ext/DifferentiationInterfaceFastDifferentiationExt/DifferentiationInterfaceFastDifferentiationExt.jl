module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoFastDifferentiation
import DifferentiationInterface as DI
using FastDifferentiation:
    derivative,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.mode(::AutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.supports_mutation(::AutoFastDifferentiation) = DI.MutationNotSupported()
DI.pullback_performance(::AutoFastDifferentiation) = DI.PullbackSlow()

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

include("allocating.jl")

end

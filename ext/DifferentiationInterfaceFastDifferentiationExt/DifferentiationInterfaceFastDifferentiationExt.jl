module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoFastDifferentiation
using DifferentiationInterface: myupdate!!, myvec
import DifferentiationInterface as DI
using FastDifferentiation:
    derivative,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables
using LinearAlgebra: dot
using RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.mode(::AutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.supports_mutation(::AutoFastDifferentiation) = DI.MutationNotSupported()
DI.supports_pullback(::AutoFastDifferentiation) = DI.PullbackNotSupported()

include("allocating.jl")

end

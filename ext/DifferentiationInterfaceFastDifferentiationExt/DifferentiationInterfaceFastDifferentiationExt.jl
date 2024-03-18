module DifferentiationInterfaceFastDifferentiationExt

using DifferentiationInterface: AutoFastDifferentiation
import DifferentiationInterface as DI
using DocStringExtensions
using FastDifferentiation:
    derivative,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables
using LinearAlgebra: dot
using RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.mutation_behavior(::AutoFastDifferentiation) = DI.MutationNotSupported()

include("allocating.jl")

end

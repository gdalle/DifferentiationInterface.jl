module DifferentiationInterfaceFiniteDiffExt

using ADTypes: AutoFiniteDiff
import DifferentiationInterface as DI
using FiniteDiff:
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_jacobian,
    finite_difference_jacobian!
using LinearAlgebra: dot, mul!

# see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions
const FUNCTION_INPLACE = Val{true}
const FUNCTION_NOT_INPLACE = Val{false}

include("allocating.jl")
include("mutating.jl")

end # module

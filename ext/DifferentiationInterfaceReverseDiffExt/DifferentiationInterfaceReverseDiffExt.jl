module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using LinearAlgebra: mul!
using ReverseDiff: gradient, gradient!, jacobian, jacobian!

include("allocating.jl")
include("mutating.jl")

end # module

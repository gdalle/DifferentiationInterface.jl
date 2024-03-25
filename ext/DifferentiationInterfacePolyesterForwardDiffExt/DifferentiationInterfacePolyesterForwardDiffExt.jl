module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoPolyesterForwardDiff, AutoForwardDiff
using DifferentiationInterface: mysimilar
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Chunk
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!

include("allocating.jl")
include("mutating.jl")

end # module

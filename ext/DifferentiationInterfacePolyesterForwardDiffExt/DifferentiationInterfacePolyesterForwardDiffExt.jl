module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoPolyesterForwardDiff, AutoForwardDiff
import DifferentiationInterface as DI
using DocStringExtensions
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

include("allocating.jl")
include("mutating.jl")

end # module

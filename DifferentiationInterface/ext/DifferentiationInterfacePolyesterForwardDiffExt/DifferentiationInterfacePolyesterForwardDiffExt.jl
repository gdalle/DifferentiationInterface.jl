module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes:
    AutoForwardDiff,
    AutoPolyesterForwardDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoHessianExtras,
    NoJacobianExtras,
    PushforwardExtras
using DocStringExtensions
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

DI.check_available(::AutoPolyesterForwardDiff) = true

function single_threaded(::AutoPolyesterForwardDiff{C}) where {C}
    return AutoForwardDiff{C,Nothing}(nothing)
end

function single_threaded(::AutoSparsePolyesterForwardDiff{C}) where {C}
    return AutoSparseForwardDiff{C,Nothing}(nothing)
end

include("onearg.jl")
include("twoarg.jl")

end # module

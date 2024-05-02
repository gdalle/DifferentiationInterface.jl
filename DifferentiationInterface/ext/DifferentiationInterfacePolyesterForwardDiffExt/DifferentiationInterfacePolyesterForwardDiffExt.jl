module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
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
using Compat: @compat

DI.check_available(::AutoPolyesterForwardDiff) = true

function single_threaded(backend::AutoPolyesterForwardDiff{C,T}) where {C,T}
    return AutoForwardDiff{C,T}(backend.tag)
end

include("onearg.jl")
include("twoarg.jl")

end # module

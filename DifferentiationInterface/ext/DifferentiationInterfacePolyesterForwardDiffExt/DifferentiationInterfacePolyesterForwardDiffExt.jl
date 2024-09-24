module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    JacobianPrep,
    NoDerivativePrep,
    NoGradientPrep,
    NoHessianPrep,
    NoJacobianPrep,
    PushforwardPrep,
    SecondDerivativePrep,
    Tangents
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

DI.check_available(::AutoPolyesterForwardDiff) = true

function single_threaded(backend::AutoPolyesterForwardDiff{C,T}) where {C,T}
    return AutoForwardDiff{C,T}(backend.tag)
end

include("onearg.jl")
include("twoarg.jl")

end # module

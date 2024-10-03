module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    Context,
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    NoDerivativePrep,
    NoGradientPrep,
    NoHessianPrep,
    NoJacobianPrep,
    PushforwardPrep,
    SecondDerivativePrep,
    unwrap,
    with_contexts
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

function single_threaded(backend::AutoPolyesterForwardDiff{C,T}) where {C,T}
    return AutoForwardDiff{C,T}(backend.tag)
end

DI.check_available(::AutoPolyesterForwardDiff) = true

function DI.pick_batchsize(backend::AutoPolyesterForwardDiff, dimension::Integer)
    return DI.pick_batchsize(single_threaded(backend), dimension)
end

function DI.threshold_batchsize(
    backend::AutoPolyesterForwardDiff{C1}, C2::Integer
) where {C1}
    C = (C1 === nothing) ? nothing : min(C1, C2)
    return AutoPolyesterForwardDiff(; chunksize=C, tag=backend.tag)
end

include("onearg.jl")
include("twoarg.jl")

end # module

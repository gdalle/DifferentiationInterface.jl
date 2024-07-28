module DifferentiationInterfaceForwardDiffReverseDiffExt

using ADTypes: AutoForwardDiff, AutoReverseDiff
using Compat: @compat
import DifferentiationInterface as DI
using DifferentiationInterface:
    Batch,
    HVPExtras,
    SecondOrder,
    gradient,
    hvp,
    inner,
    outer,
    prepare_gradient,
    prepare_hvp,
    prepare_hvp_batched,
    prepare_pushforward,
    prepare_pushforward_batched,
    PreparedInnerGradient
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

DIForwardDiffExt = Base.get_extension(DI, :DifferentiationInterfaceForwardDiffExt)

struct DIForwardDiffOverReverseDiffTag{F} end

struct ForwardDiffOverReverseDiffHVPExtras{IG} <: HVPExtras
    inner_gradient::IG
end

function _hvp_tag(f::F, backend::AutoForwardDiff, x) where {F}
    if isnothing(backend.tag)
        return ForwardDiff.Tag(DIForwardDiffOverReverseDiffTag{F}, eltype(x))
    else
        return backend.tag
    end
end

function DI.prepare_hvp(
    f::F, backend::SecondOrder{<:AutoForwardDiff,<:AutoReverseDiff}, x, dx
) where {F}
    T = _hvp_tag(f, outer(backend), x)
    xdual = DIForwardDiffExt.make_dual(T, x, dx)
    tape = ReverseDiff.GradientTape(f, xdual)
    if inner(backend) isa AutoReverseDiff{true}
        tape = ReverseDiff.compile(tape)
    end
    inner_gradient(x) = ReverseDiff.gradient!(tape, x)
    return ForwardDiffOverReverseDiffHVPExtras(inner_gradient)
end

function DI.hvp(
    f::F,
    ::SecondOrder{<:AutoForwardDiff,<:AutoReverseDiff},
    x,
    dx,
    extras::ForwardDiffOverReverseDiffHVPExtras,
) where {F}
    @compat (; inner_gradient) = extras
    T = _hvp_tag(f, outer(backend), x)
    xdual = DIForwardDiffExt.make_dual(T, x, dx)
    ydual = inner_gradient(xdual)
    return DIForwardDiffExt.myderivative(T, ydual)
end

function DI.hvp!(
    f::F,
    dg,
    ::SecondOrder{<:AutoForwardDiff,<:AutoReverseDiff},
    x,
    dx,
    extras::ForwardDiffOverReverseDiffHVPExtras,
) where {F}
    @compat (; inner_gradient) = extras
    T = _hvp_tag(f, outer(backend), x)
    xdual = DIForwardDiffExt.make_dual(T, x, dx)
    ydual = inner_gradient(xdual)
    DIForwardDiffExt.myderivative!(T, dg, ydual)
    return dg
end

end

module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface:
    Constant, NoGradientPrep, NoPullbackPrep, PullbackPrep, unwrap, with_contexts
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.check_available(::AutoTracker) = true
DI.inplace_support(::AutoTracker) = DI.InPlaceNotSupported()

## Pullback

struct TrackerPullbackPrepSamePoint{Y,PB} <: PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f, ::AutoTracker, x, ty::NTuple, contexts::Vararg{Constant,C}
) where {C}
    return NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f, ::NoPullbackPrep, ::AutoTracker, x, ty::NTuple, contexts::Vararg{Constant,C}
) where {C}
    y, pb = forward(f, x, map(unwrap, contexts)...)
    return TrackerPullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, ::NoPullbackPrep, ::AutoTracker, x, ty::NTuple, contexts::Vararg{Constant,C}
) where {C}
    y, pb = forward(f, x, map(unwrap, contexts)...)
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::TrackerPullbackPrepSamePoint,
    ::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    (; y, pb) = prep
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::TrackerPullbackPrepSamePoint,
    ::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    (; pb) = prep
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return tx
end

## Gradient

function DI.prepare_gradient(f, ::AutoTracker, x, contexts::Vararg{Constant,C}) where {C}
    return NoGradientPrep()
end

function DI.value_and_gradient(
    f, ::NoGradientPrep, ::AutoTracker, x, contexts::Vararg{Constant,C}
) where {C}
    (; val, grad) = withgradient(f, x, map(unwrap, contexts)...)
    return val, data(first(grad))
end

function DI.gradient(
    f, ::NoGradientPrep, ::AutoTracker, x, contexts::Vararg{Constant,C}
) where {C}
    (; grad) = withgradient(f, x, map(unwrap, contexts)...)
    return data(first(grad))
end

function DI.value_and_gradient!(
    f, grad, prep::NoGradientPrep, backend::AutoTracker, x, contexts::Vararg{Constant,C}
) where {C}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f, grad, prep::NoGradientPrep, backend::AutoTracker, x, contexts::Vararg{Constant,C}
) where {C}
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

end

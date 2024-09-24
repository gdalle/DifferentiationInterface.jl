module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: NoGradientPrep, NoPullbackPrep, PullbackPrep, Tangents
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient
using Compat

DI.check_available(::AutoTracker) = true
DI.inplace_support(::AutoTracker) = DI.InPlaceNotSupported()

## Pullback

struct TrackerPullbackPrepSamePoint{Y,PB} <: PullbackPrep
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoTracker, x, ty::Tangents) = NoPullbackPrep()

function DI.prepare_pullback_same_point(f, ::NoPullbackPrep, ::AutoTracker, x, ty::Tangents)
    y, pb = forward(f, x)
    return TrackerPullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(f, ::NoPullbackPrep, ::AutoTracker, x, ty::Tangents)
    y, pb = forward(f, x)
    tx = map(ty) do dy
        data(only(pb(dy)))
    end
    return y, tx
end

function DI.value_and_pullback(
    f, prep::TrackerPullbackPrepSamePoint, ::AutoTracker, x, ty::Tangents
)
    @compat (; y, pb) = prep
    tx = map(ty) do dy
        data(only(pb(dy)))
    end
    return copy(y), tx
end

function DI.pullback(f, prep::TrackerPullbackPrepSamePoint, ::AutoTracker, x, ty::Tangents)
    @compat (; pb) = prep
    tx = map(ty) do dy
        data(only(pb(dy)))
    end
    return tx
end

## Gradient

DI.prepare_gradient(f, ::AutoTracker, x) = NoGradientPrep()

function DI.value_and_gradient(f, ::NoGradientPrep, ::AutoTracker, x)
    @compat (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.gradient(f, ::NoGradientPrep, ::AutoTracker, x)
    @compat (; grad) = withgradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient!(f, grad, prep::NoGradientPrep, backend::AutoTracker, x)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, prep::NoGradientPrep, backend::AutoTracker, x)
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

end

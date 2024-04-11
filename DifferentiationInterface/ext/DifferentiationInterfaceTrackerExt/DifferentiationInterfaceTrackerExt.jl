module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: NoGradientExtras, NoPullbackExtras
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

DI.prepare_pullback(f, ::AutoTracker, x) = NoPullbackExtras()

function DI.value_and_pullback_split(f, ::AutoTracker, x, ::NoPullbackExtras)
    y, back = forward(f, x)
    pullbackfunc(dy) = data(only(back(dy)))
    return y, pullbackfunc
end

function DI.value_and_pullback!_split(f, backend::AutoTracker, x, ::NoPullbackExtras)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    pullbackfunc!(dx, dy) = copyto!(dx, pullbackfunc(dy))
    return y, pullbackfunc!
end

function DI.value_and_pullback(f, backend::AutoTracker, x, dy, extras::NoPullbackExtras)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    return y, pullbackfunc(dy)
end

## Gradient

DI.prepare_gradient(f, ::AutoTracker, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    (; grad) = withgradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

end

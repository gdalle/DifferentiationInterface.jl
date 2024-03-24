module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f::F, ::AutoTracker, x, dy, extras::Nothing) where {F}
    y, back = forward(f, x)
    return y, data(only(back(dy)))
end

## Gradient

function DI.value_and_gradient(f::F, ::AutoTracker, x, extras::Nothing) where {F}
    (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f::F, ::AutoTracker, x, extras::Nothing) where {F}
    return only(gradient(f, x))
end

function DI.value_and_gradient!!(
    f::F, grad, backend::AutoTracker, x, extras::Nothing
) where {F}
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f::F, grad, backend::AutoTracker, x, extras::Nothing) where {F}
    return DI.gradient(f, backend, x, extras)
end

end

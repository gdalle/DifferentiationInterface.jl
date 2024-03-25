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

function DI.gradient(f::F, ::AutoTracker, x, extras::Nothing) where {F}
    grad = gradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient(f::F, ::AutoTracker, x, extras::Nothing) where {F}
    (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.value_and_gradient(
    f::F, backend::AutoTracker, x::Number, extras::Nothing
) where {F}
    # fix for https://github.com/FluxML/Tracker.jl/issues/165
    return f(x), DI.gradient(f, backend, x, extras)
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

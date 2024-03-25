module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f, ::AutoTracker, x, dy, extras::Nothing)
    y, back = forward(f, x)
    return y, data(only(back(dy)))
end

## Gradient

function DI.gradient(f, ::AutoTracker, x, extras::Nothing)
    grad = gradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient(f, ::AutoTracker, x, extras::Nothing)
    (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.value_and_gradient(f, backend::AutoTracker, x::Number, extras::Nothing)
    # fix for https://github.com/FluxML/Tracker.jl/issues/165
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.value_and_gradient!!(f, grad, backend::AutoTracker, x, extras::Nothing)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AutoTracker, x, extras::Nothing)
    return DI.gradient(f, backend, x, extras)
end

end

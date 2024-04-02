module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: NoGradientExtras, NoPullbackExtras
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

DI.prepare_pullback(f, ::AutoTracker, x) = NoPullbackExtras()

function DI.value_and_pullback(f, ::AutoTracker, x, dy, ::NoPullbackExtras)
    y, back = forward(f, x)
    return y, data(only(back(dy)))
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

function DI.value_and_gradient(f, ::AutoTracker, x::Number, ::NoGradientExtras)
    # fix for https://github.com/FluxML/Tracker.jl/issues/165
    return f(x), data(only(gradient(f, x)))
end

function DI.gradient(f, ::AutoTracker, x::Number, ::NoGradientExtras)
    # fix for https://github.com/FluxML/Tracker.jl/issues/165
    return data(only(gradient(f, x)))
end

function DI.value_and_gradient!!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return DI.gradient(f, backend, x, extras)
end

end

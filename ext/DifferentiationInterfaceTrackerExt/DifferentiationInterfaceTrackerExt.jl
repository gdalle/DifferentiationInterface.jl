module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: update!
using Tracker: Tracker, back, forward, gradient, jacobian, param, withgradient

## Pullback

function DI.value_and_pullback(::AutoTracker, f, x, dy, extras::Nothing)
    y, back = forward(f, x)
    return y, Tracker.data(only(back(dy)))
end

function DI.value_and_pullback!(dx, backend::AutoTracker, f, x, dy, extras::Nothing)
    y, new_dx = DI.value_and_pullback(backend, f, x, dy, extras)
    return y, update!(dx, new_dx)
end

## Gradient

function DI.value_and_gradient(::AutoTracker, f, x::AbstractArray, extras::Nothing)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.gradient(::AutoTracker, f, x::AbstractArray, extras::Nothing)
    return only(gradient(f, x))
end

function DI.value_and_gradient!(
    grad::AbstractArray, backend::AutoTracker, f, x::AbstractArray, extras::Nothing
)
    y, new_grad = DI.value_and_gradient(backend, f, x, extras)
    grad .= Tracker.data(new_grad)
    return y, grad
end

function DI.gradient!(
    grad::AbstractArray, backend::AutoTracker, f, x::AbstractArray, extras::Nothing
)
    new_grad = DI.gradient(backend, f, x, extras)
    grad .= Tracker.data(new_grad)
    return grad
end

end

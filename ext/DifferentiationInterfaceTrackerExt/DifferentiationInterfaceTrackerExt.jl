module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using Tracker: Tracker, back!, gradient, param, withgradient

# TODO: make faster by prerecording tape

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

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoTracker,
    f,
    x::AbstractArray,
    dy::Number,
    extras::Nothing,
)
    y, dx = DI.value_and_gradient!(dx, backend, f, x, extras)
    dx .*= dy
    return y, dx
end

end

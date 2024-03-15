## Primitives

function DI.value_and_pullback!(
    dx::AbstractArray,
    ::AutoReverseDiff,
    f,
    x::AbstractArray,
    dy::Real,
    extras::Nothing=nothing,
)
    res = DiffResults.DiffResult(zero(dy), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    ::AutoReverseDiff,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    jac = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback!(
    _dx::Number, backend::AutoReverseDiff, f, x::Number, dy, extras::Nothing=nothing
)
    x_array = [x]
    dx_array = similar(x_array)
    y, dx_array = DI.value_and_pullback!(dx_array, backend, f ∘ only, x_array, dy, extras)
    return y, only(dx_array)
end

## Utilities (TODO: use DiffResults)

function DI.value_and_gradient(
    ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    grad = gradient(f, x)
    return y, grad
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    gradient!(grad, f, x)
    return y, grad
end

function DI.value_and_jacobian(
    ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    jac = jacobian(f, x)
    return y, jac
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    jacobian!(jac, f, x)
    return y, jac
end

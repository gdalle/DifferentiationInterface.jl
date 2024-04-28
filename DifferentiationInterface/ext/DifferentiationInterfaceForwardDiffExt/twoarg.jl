## Pushforward

struct ForwardDiffTwoArgPushforwardExtras{T,X,Y} <: PushforwardExtras
    xdual_tmp::X
    ydual_tmp::Y
end

function DI.prepare_pushforward(f!, y, ::AutoForwardDiff, x, dx)
    T = tag_type(f!, x)
    xdual_tmp = make_dual(T, x, dx)
    ydual_tmp = make_dual(T, y, similar(y))
    return ForwardDiffTwoArgPushforwardExtras{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function compute_ydual_twoarg(
    f!, y, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {T}
    (; xdual_tmp, ydual_tmp) = extras
    xdual_tmp = if x isa Number
        make_dual(T, x, dx)
    else
        make_dual!(T, xdual_tmp, x, dx)
    end
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!, y, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myvalue!(T, y, ydual_tmp)
    dy = myderivative(T, ydual_tmp)
    return y, dy
end

function DI.pushforward(
    f!, y, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    dy = myderivative(T, ydual_tmp)
    return dy
end

function DI.value_and_pushforward!(
    f!, y, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myvalue!(T, y, ydual_tmp)
    myderivative!(T, dy, ydual_tmp)
    return y, dy
end

function DI.pushforward!(
    f!, y, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myderivative!(T, dy, ydual_tmp)
    return dy
end

## Derivative

struct ForwardDiffTwoArgDerivativeExtras{C} <: DerivativeExtras
    config::C
end

function DI.prepare_derivative(f!, y::AbstractArray, ::AutoForwardDiff, x::Number)
    return ForwardDiffTwoArgDerivativeExtras(DerivativeConfig(f!, y, x))
end

function DI.value_and_derivative(
    f!,
    y::AbstractArray,
    ::AutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!,
    y::AbstractArray,
    ::AutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    der = derivative(f!, y, x, extras.config)
    return der
end

function DI.derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    der = derivative!(der, f!, y, x, extras.config)
    return der
end

## Jacobian

struct ForwardDiffTwoArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(
    f!, y::AbstractArray, backend::AutoForwardDiff, x::AbstractArray
)
    return ForwardDiffTwoArgJacobianExtras(
        JacobianConfig(f!, y, x, choose_chunk(backend, x))
    )
end

function DI.value_and_jacobian(
    f!,
    y::AbstractArray,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!,
    y::AbstractArray,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = jacobian(f!, y, x, extras.config)
    return jac
end

function DI.jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = jacobian!(jac, f!, y, x, extras.config)
    return jac
end

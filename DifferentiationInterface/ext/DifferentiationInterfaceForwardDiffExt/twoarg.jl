## Pushforward

struct ForwardDiffTwoArgPushforwardExtras{T,X,Y} <: PushforwardExtras
    xdual_tmp::X
    ydual_tmp::Y
end

function DI.prepare_pushforward(f!::F, y, backend::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, dx)
    ydual_tmp = make_dual_similar(T, y, dx)  # dx only for batch size
    return ForwardDiffTwoArgPushforwardExtras{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function DI.prepare_pushforward_batched(
    f!::F, y, backend::AutoForwardDiff, x, dx::Batch
) where {F}
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, dx)
    ydual_tmp = make_dual_similar(T, y, dx)  # dx only for batch size
    return ForwardDiffTwoArgPushforwardExtras{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function compute_ydual_twoarg(
    f!::F, y, x::Number, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    @compat (; ydual_tmp) = extras
    xdual_tmp = make_dual(T, x, dx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F, y, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    @compat (; xdual_tmp, ydual_tmp) = extras
    make_dual!(T, xdual_tmp, x, dx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F, y, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myvalue!(T, y, ydual_tmp)
    dy = myderivative(T, ydual_tmp)
    return y, dy
end

function DI.pushforward(
    f!::F, y, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    dy = myderivative(T, ydual_tmp)
    return dy
end

function DI.value_and_pushforward!(
    f!::F, y, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myvalue!(T, y, ydual_tmp)
    myderivative!(T, dy, ydual_tmp)
    return y, dy
end

function DI.pushforward!(
    f!::F, y, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    myderivative!(T, dy, ydual_tmp)
    return dy
end

function DI.pushforward_batched(
    f!::F,
    y,
    ::AutoForwardDiff,
    x,
    dx::Batch{B},
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    dy = mypartials(T, Val(B), ydual_tmp)
    return dy
end

function DI.pushforward_batched!(
    f!::F,
    y,
    dy::Batch{B},
    ::AutoForwardDiff,
    x,
    dx::Batch{B},
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, dx, extras)
    mypartials!(T, dy, ydual_tmp)
    return dy
end

## Derivative

### Unprepared

function DI.value_and_derivative(f!::F, y, ::AutoForwardDiff, x) where {F}
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, f!, y, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(f!::F, y, der, ::AutoForwardDiff, x) where {F}
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, f!, y, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(f!::F, y, ::AutoForwardDiff, x) where {F}
    return derivative(f!, y, x)
end

function DI.derivative!(f!::F, y, der, ::AutoForwardDiff, x) where {F}
    return derivative!(der, f!, y, x)
end

### Prepared

struct ForwardDiffTwoArgDerivativeExtras{C} <: DerivativeExtras
    config::C
end

function DI.prepare_derivative(f!::F, y, ::AutoForwardDiff, x) where {F}
    return ForwardDiffTwoArgDerivativeExtras(DerivativeConfig(f!, y, x))
end

function DI.value_and_derivative(
    f!::F, y, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgDerivativeExtras
) where {F}
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!::F, y, der, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgDerivativeExtras
) where {F}
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!::F, y, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgDerivativeExtras
) where {F}
    return derivative(f!, y, x, extras.config)
end

function DI.derivative!(
    f!::F, y, der, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgDerivativeExtras
) where {F}
    return derivative!(der, f!, y, x, extras.config)
end

## Jacobian

### Unprepared

function DI.value_and_jacobian(f!::F, y, ::AutoForwardDiff, x) where {F}
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(f!::F, y, jac, ::AutoForwardDiff, x) where {F}
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(f!::F, y, ::AutoForwardDiff, x) where {F}
    return jacobian(f!, y, x)
end

function DI.jacobian!(f!::F, y, jac, ::AutoForwardDiff, x) where {F}
    return jacobian!(jac, f!, y, x)
end

### Prepared

struct ForwardDiffTwoArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(f!::F, y, backend::AutoForwardDiff, x) where {F}
    return ForwardDiffTwoArgJacobianExtras(
        JacobianConfig(f!, y, x, choose_chunk(backend, x))
    )
end

function DI.value_and_jacobian(
    f!::F, y, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgJacobianExtras
) where {F}
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgJacobianExtras
) where {F}
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!::F, y, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgJacobianExtras
) where {F}
    return jacobian(f!, y, x, extras.config)
end

function DI.jacobian!(
    f!::F, y, jac, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgJacobianExtras
) where {F}
    return jacobian!(jac, f!, y, x, extras.config)
end

## Pushforward

struct ForwardDiffTwoArgPushforwardExtras{T,X,Y} <: PushforwardExtras
    xdual_tmp::X
    ydual_tmp::Y
end

function DI.prepare_pushforward(
    f!::F, y, backend::AutoForwardDiff, x, tx::Tangents
) where {F}
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    ydual_tmp = make_dual_similar(T, y, tx)  # dx only for batch size
    return ForwardDiffTwoArgPushforwardExtras{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function compute_ydual_twoarg(
    f!::F, y, extras::ForwardDiffTwoArgPushforwardExtras{T}, x::Number, tx::Tangents
) where {F,T}
    @compat (; ydual_tmp) = extras
    xdual_tmp = make_dual(T, x, tx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F, y, extras::ForwardDiffTwoArgPushforwardExtras{T}, x, tx::Tangents
) where {F,T}
    @compat (; xdual_tmp, ydual_tmp) = extras
    make_dual!(T, xdual_tmp, x, tx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F,
    y,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f, y, extras, x, tx)
    myvalue!(T, y, ydual_tmp)
    ty = mypartials(T, Val(B), ydual_tmp)
    return y, ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, extras, x, tx)
    myvalue!(T, y, ydual_tmp)
    mypartials!(T, ty, ydual_tmp)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f!, y, extras, x, tx)
    ty = mypartials(T, Val(B), ydual_tmp)
    return ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::Tangents,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, extras, x, tx)
    mypartials!(T, ty, ydual_tmp)
    return ty
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
    f!::F, y, extras::ForwardDiffTwoArgDerivativeExtras, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!::F, y, der, extras::ForwardDiffTwoArgDerivativeExtras, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!::F, y, extras::ForwardDiffTwoArgDerivativeExtras, ::AutoForwardDiff, x
) where {F}
    return derivative(f!, y, x, extras.config)
end

function DI.derivative!(
    f!::F, y, der, extras::ForwardDiffTwoArgDerivativeExtras, ::AutoForwardDiff, x
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
    f!::F, y, extras::ForwardDiffTwoArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, extras::ForwardDiffTwoArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!::F, y, extras::ForwardDiffTwoArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    return jacobian(f!, y, x, extras.config)
end

function DI.jacobian!(
    f!::F, y, jac, extras::ForwardDiffTwoArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    return jacobian!(jac, f!, y, x, extras.config)
end

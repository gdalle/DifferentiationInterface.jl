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
    f!::F, y, x::Number, tx::Tangents, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    @compat (; ydual_tmp) = extras
    xdual_tmp = make_dual(T, x, tx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F, y, x, tx::Tangents, extras::ForwardDiffTwoArgPushforwardExtras{T}
) where {F,T}
    @compat (; xdual_tmp, ydual_tmp) = extras
    make_dual!(T, xdual_tmp, x, tx)
    f!(ydual_tmp, xdual_tmp)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F,
    y,
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, tx, extras)
    myvalue!(T, y, ydual_tmp)
    ty = mypartials(T, Val(B), ydual_tmp)
    return y, ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, tx, extras)
    myvalue!(T, y, ydual_tmp)
    mypartials!(T, ty, ydual_tmp)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T,B}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, tx, extras)
    ty = mypartials(T, Val(B), ydual_tmp)
    return ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::Tangents,
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    extras::ForwardDiffTwoArgPushforwardExtras{T},
) where {F,T}
    ydual_tmp = compute_ydual_twoarg(f!, y, x, tx, extras)
    mypartials!(T, ty, ydual_tmp)
    return ty
end

## Derivative

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
    der = derivative(f!, y, x, extras.config)
    return der
end

function DI.derivative!(
    f!::F, y, der, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgDerivativeExtras
) where {F}
    der = derivative!(der, f!, y, x, extras.config)
    return der
end

## Jacobian

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
    jac = jacobian(f!, y, x, extras.config)
    return jac
end

function DI.jacobian!(
    f!::F, y, jac, ::AutoForwardDiff, x, extras::ForwardDiffTwoArgJacobianExtras
) where {F}
    jac = jacobian!(jac, f!, y, x, extras.config)
    return jac
end

## Pushforward

struct ForwardDiffTwoArgPushforwardPrep{T,X,Y} <: PushforwardPrep
    xdual_tmp::X
    ydual_tmp::Y
end

function DI.prepare_pushforward(
    f!::F, y, backend::AutoForwardDiff, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    ydual_tmp = make_dual_similar(T, y, tx)  # dx only for batch size
    return ForwardDiffTwoArgPushforwardPrep{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x::Number,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    @compat (; ydual_tmp) = prep
    xdual_tmp = make_dual(T, x, tx)
    f!(ydual_tmp, xdual_tmp, map(unwrap, contexts)...)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    @compat (; xdual_tmp, ydual_tmp) = prep
    make_dual!(T, xdual_tmp, x, tx)
    f!(ydual_tmp, xdual_tmp, map(unwrap, contexts)...)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    myvalue!(T, y, ydual_tmp)
    ty = mypartials(T, Val(B), ydual_tmp)
    return y, ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    myvalue!(T, y, ydual_tmp)
    mypartials!(T, ty, ydual_tmp)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual_tmp)
    return ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::Tangents,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
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

struct ForwardDiffTwoArgDerivativePrep{C} <: DerivativePrep
    config::C
end

function DI.prepare_derivative(f!::F, y, ::AutoForwardDiff, x) where {F}
    return ForwardDiffTwoArgDerivativePrep(DerivativeConfig(f!, y, x))
end

function DI.value_and_derivative(
    f!::F, y, prep::ForwardDiffTwoArgDerivativePrep, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, f!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!::F, y, der, prep::ForwardDiffTwoArgDerivativePrep, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, f!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!::F, y, prep::ForwardDiffTwoArgDerivativePrep, ::AutoForwardDiff, x
) where {F}
    return derivative(f!, y, x, prep.config)
end

function DI.derivative!(
    f!::F, y, der, prep::ForwardDiffTwoArgDerivativePrep, ::AutoForwardDiff, x
) where {F}
    return derivative!(der, f!, y, x, prep.config)
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

struct ForwardDiffTwoArgJacobianPrep{C} <: JacobianPrep
    config::C
end

function DI.prepare_jacobian(f!::F, y, backend::AutoForwardDiff, x) where {F}
    return ForwardDiffTwoArgJacobianPrep(JacobianConfig(f!, y, x, choose_chunk(backend, x)))
end

function DI.value_and_jacobian(
    f!::F, y, prep::ForwardDiffTwoArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!::F, y, jac, prep::ForwardDiffTwoArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!::F, y, prep::ForwardDiffTwoArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    return jacobian(f!, y, x, prep.config)
end

function DI.jacobian!(
    f!::F, y, jac, prep::ForwardDiffTwoArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    return jacobian!(jac, f!, y, x, prep.config)
end

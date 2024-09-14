## Pushforward

struct ForwardDiffOneArgPushforwardExtras{T,X} <: PushforwardExtras
    xdual_tmp::X
end

function DI.prepare_pushforward(
    f::F, backend::AutoForwardDiff, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    T = tag_type(f, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    return ForwardDiffOneArgPushforwardExtras{T,typeof(xdual_tmp)}(xdual_tmp)
end

function compute_ydual_onearg(
    f::F,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    x::Number,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    xdual_tmp = make_dual(T, x, tx)
    ydual = f(xdual_tmp, map(unwrap, contexts)...)
    return ydual
end

function compute_ydual_onearg(
    f::F,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    @compat (; xdual_tmp) = extras
    make_dual!(T, xdual_tmp, x, tx)
    ydual = f(xdual_tmp, map(unwrap, contexts)...)
    return ydual
end

function DI.value_and_pushforward(
    f::F,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, extras, x, tx, contexts...)
    y = myvalue(T, ydual)
    ty = mypartials(T, Val(B), ydual)
    return y, ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::Tangents,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, extras, x, tx, contexts...)
    y = myvalue(T, ydual)
    mypartials!(T, ty, ydual)
    return y, ty
end

function DI.pushforward(
    f::F,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, extras, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual)
    return ty
end

function DI.pushforward!(
    f::F,
    ty::Tangents,
    extras::ForwardDiffOneArgPushforwardExtras{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, extras, x, tx, contexts...)
    mypartials!(T, ty, ydual)
    return ty
end

## Derivative

struct ForwardDiffOneArgDerivativeExtras{E} <: DerivativeExtras
    pushforward_extras::E
end

function DI.prepare_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    pushforward_extras = DI.prepare_pushforward(f, backend, x, Tangents(one(x)))
    return ForwardDiffOneArgDerivativeExtras(pushforward_extras)
end

function DI.value_and_derivative(
    f::F, extras::ForwardDiffOneArgDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    y, ty = DI.value_and_pushforward(
        f, extras.pushforward_extras, backend, x, Tangents(one(x))
    )
    return y, only(ty)
end

function DI.value_and_derivative!(
    f::F, der, extras::ForwardDiffOneArgDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    y, _ = DI.value_and_pushforward!(
        f, Tangents(der), extras.pushforward_extras, backend, x, Tangents(one(x))
    )
    return y, der
end

function DI.derivative(
    f::F, extras::ForwardDiffOneArgDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    return only(DI.pushforward(f, extras.pushforward_extras, backend, x, Tangents(one(x))))
end

function DI.derivative!(
    f::F, der, extras::ForwardDiffOneArgDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    DI.pushforward!(
        f, Tangents(der), extras.pushforward_extras, backend, x, Tangents(one(x))
    )
    return der
end

## Gradient

### Unprepared

function DI.value_and_gradient!(f::F, grad, ::AutoForwardDiff, x) where {F}
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, f, x)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(f::F, ::AutoForwardDiff, x) where {F}
    result = GradientResult(x)
    result = gradient!(result, f, x)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(f::F, grad, ::AutoForwardDiff, x) where {F}
    return gradient!(grad, f, x)
end

function DI.gradient(f::F, ::AutoForwardDiff, x) where {F}
    return gradient(f, x)
end

### Prepared

struct ForwardDiffGradientExtras{C} <: GradientExtras
    config::C
end

function DI.prepare_gradient(f::F, backend::AutoForwardDiff, x::AbstractArray) where {F}
    return ForwardDiffGradientExtras(GradientConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_gradient!(
    f::F, grad, extras::ForwardDiffGradientExtras, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f::F, extras::ForwardDiffGradientExtras, ::AutoForwardDiff, x
) where {F}
    result = GradientResult(x)
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(
    f::F, grad, extras::ForwardDiffGradientExtras, ::AutoForwardDiff, x
) where {F}
    return gradient!(grad, f, x, extras.config)
end

function DI.gradient(
    f::F, extras::ForwardDiffGradientExtras, ::AutoForwardDiff, x
) where {F}
    return gradient(f, x, extras.config)
end

## Jacobian

### Unprepared

function DI.value_and_jacobian!(f::F, jac, ::AutoForwardDiff, x) where {F}
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f, x)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(f::F, ::AutoForwardDiff, x) where {F}
    return f(x), jacobian(f, x)
end

function DI.jacobian!(f::F, jac, ::AutoForwardDiff, x) where {F}
    return jacobian!(jac, f, x)
end

function DI.jacobian(f::F, ::AutoForwardDiff, x) where {F}
    return jacobian(f, x)
end

### Prepared

struct ForwardDiffOneArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(f, backend::AutoForwardDiff, x)
    return ForwardDiffOneArgJacobianExtras(JacobianConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_jacobian!(
    f::F, jac, extras::ForwardDiffOneArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f::F, extras::ForwardDiffOneArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    return f(x), jacobian(f, x, extras.config)
end

function DI.jacobian!(
    f::F, jac, extras::ForwardDiffOneArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    return jacobian!(jac, f, x, extras.config)
end

function DI.jacobian(
    f::F, extras::ForwardDiffOneArgJacobianExtras, ::AutoForwardDiff, x
) where {F}
    return jacobian(f, x, extras.config)
end

## Second derivative

function DI.prepare_second_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    return NoSecondDerivativeExtras()
end

function DI.second_derivative(
    f::F, ::NoSecondDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative(T, myderivative(T2, ydual))
end

function DI.second_derivative!(
    f::F, der2, ::NoSecondDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative!(T, der2, myderivative(T2, ydual))
end

function DI.value_derivative_and_second_derivative(
    f::F, ::NoSecondDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    y = myvalue(T, myvalue(T2, ydual))
    der = myderivative(T, myvalue(T2, ydual))
    der2 = myderivative(T, myderivative(T2, ydual))
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f::F, der, der2, ::NoSecondDerivativeExtras, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    y = myvalue(T, myvalue(T2, ydual))
    myderivative!(T, der, myvalue(T2, ydual))
    myderivative!(T, der2, myderivative(T2, ydual))
    return y, der, der2
end

## Hessian

### Unprepared

function DI.hessian!(f::F, hess, ::AutoForwardDiff, x) where {F}
    return hessian!(hess, f, x)
end

function DI.hessian(f::F, ::AutoForwardDiff, x) where {F}
    return hessian(f, x)
end

function DI.value_gradient_and_hessian!(f::F, grad, hess, ::AutoForwardDiff, x) where {F}
    result = MutableDiffResult(one(eltype(x)), (grad, hess))
    result = hessian!(result, f, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(f::F, ::AutoForwardDiff, x) where {F}
    result = HessianResult(x)
    result = hessian!(result, f, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

### Prepared

struct ForwardDiffHessianExtras{C1,C2,C3} <: HessianExtras
    array_config::C1
    manual_result_config::C2
    auto_result_config::C3
end

function DI.prepare_hessian(f, backend::AutoForwardDiff, x)
    manual_result = MutableDiffResult(
        one(eltype(x)), (similar(x), similar(x, length(x), length(x)))
    )
    auto_result = HessianResult(x)
    chunk = choose_chunk(backend, x)
    array_config = HessianConfig(f, x, chunk)
    manual_result_config = HessianConfig(f, manual_result, x, chunk)
    auto_result_config = HessianConfig(f, auto_result, x, chunk)
    return ForwardDiffHessianExtras(array_config, manual_result_config, auto_result_config)
end

function DI.hessian!(
    f::F, hess, extras::ForwardDiffHessianExtras, ::AutoForwardDiff, x
) where {F}
    return hessian!(hess, f, x, extras.array_config)
end

function DI.hessian(f::F, extras::ForwardDiffHessianExtras, ::AutoForwardDiff, x) where {F}
    return hessian(f, x, extras.array_config)
end

function DI.value_gradient_and_hessian!(
    f::F, grad, hess, extras::ForwardDiffHessianExtras, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(one(eltype(x)), (grad, hess))
    result = hessian!(result, f, x, extras.manual_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f::F, extras::ForwardDiffHessianExtras, ::AutoForwardDiff, x
) where {F}
    result = HessianResult(x)
    result = hessian!(result, f, x, extras.auto_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

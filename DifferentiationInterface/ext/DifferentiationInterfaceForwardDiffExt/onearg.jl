## Pushforward

struct ForwardDiffOneArgPushforwardPrep{T,X} <: PushforwardPrep
    xdual_tmp::X
end

function DI.prepare_pushforward(
    f::F, backend::AutoForwardDiff, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    T = tag_type(f, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    return ForwardDiffOneArgPushforwardPrep{T,typeof(xdual_tmp)}(xdual_tmp)
end

function compute_ydual_onearg(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
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
    prep::ForwardDiffOneArgPushforwardPrep{T},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    @compat (; xdual_tmp) = prep
    make_dual!(T, xdual_tmp, x, tx)
    ydual = f(xdual_tmp, map(unwrap, contexts)...)
    return ydual
end

function DI.value_and_pushforward(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    y = myvalue(T, ydual)
    ty = mypartials(T, Val(B), ydual)
    return y, ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::Tangents,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    y = myvalue(T, ydual)
    mypartials!(T, ty, ydual)
    return y, ty
end

function DI.pushforward(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual)
    return ty
end

function DI.pushforward!(
    f::F,
    ty::Tangents,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    mypartials!(T, ty, ydual)
    return ty
end

## Derivative

struct ForwardDiffOneArgDerivativePrep{E} <: DerivativePrep
    pushforward_prep::E
end

function DI.prepare_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    pushforward_prep = DI.prepare_pushforward(f, backend, x, Tangents(one(x)))
    return ForwardDiffOneArgDerivativePrep(pushforward_prep)
end

function DI.value_and_derivative(
    f::F, prep::ForwardDiffOneArgDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    y, ty = DI.value_and_pushforward(f, prep.pushforward_prep, backend, x, Tangents(one(x)))
    return y, only(ty)
end

function DI.value_and_derivative!(
    f::F, der, prep::ForwardDiffOneArgDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    y, _ = DI.value_and_pushforward!(
        f, Tangents(der), prep.pushforward_prep, backend, x, Tangents(one(x))
    )
    return y, der
end

function DI.derivative(
    f::F, prep::ForwardDiffOneArgDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    return only(DI.pushforward(f, prep.pushforward_prep, backend, x, Tangents(one(x))))
end

function DI.derivative!(
    f::F, der, prep::ForwardDiffOneArgDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    DI.pushforward!(f, Tangents(der), prep.pushforward_prep, backend, x, Tangents(one(x)))
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

struct ForwardDiffGradientPrep{C} <: GradientPrep
    config::C
end

function DI.prepare_gradient(f::F, backend::AutoForwardDiff, x::AbstractArray) where {F}
    return ForwardDiffGradientPrep(GradientConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_gradient!(
    f::F, grad, prep::ForwardDiffGradientPrep, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, f, x, prep.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f::F, prep::ForwardDiffGradientPrep, ::AutoForwardDiff, x
) where {F}
    result = GradientResult(x)
    result = gradient!(result, f, x, prep.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(
    f::F, grad, prep::ForwardDiffGradientPrep, ::AutoForwardDiff, x
) where {F}
    return gradient!(grad, f, x, prep.config)
end

function DI.gradient(f::F, prep::ForwardDiffGradientPrep, ::AutoForwardDiff, x) where {F}
    return gradient(f, x, prep.config)
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

struct ForwardDiffOneArgJacobianPrep{C} <: JacobianPrep
    config::C
end

function DI.prepare_jacobian(f, backend::AutoForwardDiff, x)
    return ForwardDiffOneArgJacobianPrep(JacobianConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_jacobian!(
    f::F, jac, prep::ForwardDiffOneArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f, x, prep.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f::F, prep::ForwardDiffOneArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    return f(x), jacobian(f, x, prep.config)
end

function DI.jacobian!(
    f::F, jac, prep::ForwardDiffOneArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    return jacobian!(jac, f, x, prep.config)
end

function DI.jacobian(
    f::F, prep::ForwardDiffOneArgJacobianPrep, ::AutoForwardDiff, x
) where {F}
    return jacobian(f, x, prep.config)
end

## Second derivative

function DI.prepare_second_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    return NoSecondDerivativePrep()
end

function DI.second_derivative(
    f::F, ::NoSecondDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative(T, myderivative(T2, ydual))
end

function DI.second_derivative!(
    f::F, der2, ::NoSecondDerivativePrep, backend::AutoForwardDiff, x
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative!(T, der2, myderivative(T2, ydual))
end

function DI.value_derivative_and_second_derivative(
    f::F, ::NoSecondDerivativePrep, backend::AutoForwardDiff, x
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
    f::F, der, der2, ::NoSecondDerivativePrep, backend::AutoForwardDiff, x
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

## HVP

function DI.prepare_hvp(f::F, backend::AutoForwardDiff, x, tx::Tangents) where {F}
    return DI.prepare_hvp(f, SecondOrder(backend, backend), x, tx)
end

function DI.hvp(f::F, prep::HVPPrep, backend::AutoForwardDiff, x, tx::Tangents) where {F}
    return DI.hvp(f, prep, SecondOrder(backend, backend), x, tx)
end

function DI.hvp!(
    f::F, tg::Tangents, prep::HVPPrep, backend::AutoForwardDiff, x, tx::Tangents
) where {F}
    return DI.hvp!(f, tg, prep, SecondOrder(backend, backend), x, tx)
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

struct ForwardDiffHessianPrep{C1,C2,C3} <: HessianPrep
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
    return ForwardDiffHessianPrep(array_config, manual_result_config, auto_result_config)
end

function DI.hessian!(
    f::F, hess, prep::ForwardDiffHessianPrep, ::AutoForwardDiff, x
) where {F}
    return hessian!(hess, f, x, prep.array_config)
end

function DI.hessian(f::F, prep::ForwardDiffHessianPrep, ::AutoForwardDiff, x) where {F}
    return hessian(f, x, prep.array_config)
end

function DI.value_gradient_and_hessian!(
    f::F, grad, hess, prep::ForwardDiffHessianPrep, ::AutoForwardDiff, x
) where {F}
    result = MutableDiffResult(one(eltype(x)), (grad, hess))
    result = hessian!(result, f, x, prep.manual_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f::F, prep::ForwardDiffHessianPrep, ::AutoForwardDiff, x
) where {F}
    result = HessianResult(x)
    result = hessian!(result, f, x, prep.auto_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

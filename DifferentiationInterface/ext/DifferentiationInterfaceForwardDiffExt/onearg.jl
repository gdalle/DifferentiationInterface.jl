## Pushforward

struct ForwardDiffOneArgPushforwardExtras{T,X} <: PushforwardExtras
    xdual_tmp::X
end

function DI.prepare_pushforward(f::F, backend::AutoForwardDiff, x, tx::Tangents) where {F}
    T = tag_type(f, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    return ForwardDiffOneArgPushforwardExtras{T,typeof(xdual_tmp)}(xdual_tmp)
end

function compute_ydual_onearg(
    f::F, x::Number, tx::Tangents, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    xdual_tmp = make_dual(T, x, tx)
    ydual = f(xdual_tmp)
    return ydual
end

function compute_ydual_onearg(
    f::F, x, tx::Tangents, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    @compat (; xdual_tmp) = extras
    make_dual!(T, xdual_tmp, x, tx)
    ydual = f(xdual_tmp)
    return ydual
end

function DI.value_and_pushforward(
    f::F,
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    extras::ForwardDiffOneArgPushforwardExtras{T},
) where {F,T,B}
    ydual = compute_ydual_onearg(f, x, tx, extras)
    y = myvalue(T, ydual)
    ty = mypartials(T, Val(B), ydual)
    return y, ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::Tangents,
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    extras::ForwardDiffOneArgPushforwardExtras{T},
) where {F,T}
    ydual = compute_ydual_onearg(f, x, tx, extras)
    y = myvalue(T, ydual)
    mypartials!(T, ty, ydual)
    return y, ty
end

function DI.pushforward(
    f::F,
    ::AutoForwardDiff,
    x,
    tx::Tangents{B},
    extras::ForwardDiffOneArgPushforwardExtras{T},
) where {F,T,B}
    ydual = compute_ydual_onearg(f, x, tx, extras)
    ty = mypartials(T, Val(B), ydual)
    return ty
end

function DI.pushforward!(
    f::F,
    ty::Tangents,
    ::AutoForwardDiff,
    x,
    tx::Tangents,
    extras::ForwardDiffOneArgPushforwardExtras{T},
) where {F,T}
    ydual = compute_ydual_onearg(f, x, tx, extras)
    mypartials!(T, ty, ydual)
    return ty
end

## Gradient

struct ForwardDiffGradientExtras{C} <: GradientExtras
    config::C
end

function DI.prepare_gradient(f::F, backend::AutoForwardDiff, x::AbstractArray) where {F}
    return ForwardDiffGradientExtras(GradientConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_gradient!(
    f::F, grad, ::AutoForwardDiff, x, extras::ForwardDiffGradientExtras
) where {F}
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffGradientExtras
) where {F}
    result = GradientResult(x)
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(
    f::F, grad, ::AutoForwardDiff, x, extras::ForwardDiffGradientExtras
) where {F}
    return gradient!(grad, f, x, extras.config)
end

function DI.gradient(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffGradientExtras
) where {F}
    return gradient(f, x, extras.config)
end

## Jacobian

struct ForwardDiffOneArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(f, backend::AutoForwardDiff, x)
    return ForwardDiffOneArgJacobianExtras(JacobianConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_jacobian!(
    f::F, jac, ::AutoForwardDiff, x, extras::ForwardDiffOneArgJacobianExtras
) where {F}
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffOneArgJacobianExtras
) where {F}
    return f(x), jacobian(f, x, extras.config)
end

function DI.jacobian!(
    f::F, jac, ::AutoForwardDiff, x, extras::ForwardDiffOneArgJacobianExtras
) where {F}
    return jacobian!(jac, f, x, extras.config)
end

function DI.jacobian(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffOneArgJacobianExtras
) where {F}
    return jacobian(f, x, extras.config)
end

## Second derivative

function DI.prepare_second_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    return NoSecondDerivativeExtras()
end

function DI.second_derivative(
    f::F, backend::AutoForwardDiff, x, ::NoSecondDerivativeExtras
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative(T, myderivative(T2, ydual))
end

function DI.second_derivative!(
    f::F, der2, backend::AutoForwardDiff, x, ::NoSecondDerivativeExtras
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    ydual = f(make_dual(T2, xdual, one(xdual)))
    return myderivative!(T, der2, myderivative(T2, ydual))
end

function DI.value_derivative_and_second_derivative(
    f::F, backend::AutoForwardDiff, x, ::NoSecondDerivativeExtras
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
    f::F, der, der2, backend::AutoForwardDiff, x, ::NoSecondDerivativeExtras
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
    f::F, hess, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras
) where {F}
    return hessian!(hess, f, x, extras.array_config)
end

function DI.hessian(f::F, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras) where {F}
    return hessian(f, x, extras.array_config)
end

function DI.value_gradient_and_hessian!(
    f::F, grad, hess, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras
) where {F}
    result = MutableDiffResult(one(eltype(x)), (grad, hess))
    result = hessian!(result, f, x, extras.manual_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras
) where {F}
    result = HessianResult(x)
    result = hessian!(result, f, x, extras.auto_result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

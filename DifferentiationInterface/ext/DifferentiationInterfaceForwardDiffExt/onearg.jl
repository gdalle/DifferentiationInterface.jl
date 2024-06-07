## Pushforward

struct ForwardDiffOneArgPushforwardExtras{T,X} <: PushforwardExtras
    xdual_tmp::X
end

function DI.prepare_pushforward(f::F, backend::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, backend, x)
    xdual_tmp = make_dual_similar(T, x)
    return ForwardDiffOneArgPushforwardExtras{T,typeof(xdual_tmp)}(xdual_tmp)
end

function compute_ydual_onearg(
    f::F, x::Number, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    xdual_tmp = make_dual(T, x, dx)
    ydual = f(xdual_tmp)
    return ydual
end

function compute_ydual_onearg(
    f::F, x, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    @compat (; xdual_tmp) = extras
    make_dual!(T, xdual_tmp, x, dx)
    ydual = f(xdual_tmp)
    return ydual
end

function DI.value_and_pushforward(
    f::F, ::AutoForwardDiff, x, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    ydual = compute_ydual_onearg(f, x, dx, extras)
    y = myvalue(T, ydual)
    new_dy = myderivative(T, ydual)
    return y, new_dy
end

function DI.pushforward(
    f::F, ::AutoForwardDiff, x, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    ydual = compute_ydual_onearg(f, x, dx, extras)
    new_dy = myderivative(T, ydual)
    return new_dy
end

function DI.value_and_pushforward!(
    f::F, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    ydual = compute_ydual_onearg(f, x, dx, extras)
    y = myvalue(T, ydual)
    myderivative!(T, dy, ydual)
    return y, dy
end

function DI.pushforward!(
    f::F, dy, ::AutoForwardDiff, x, dx, extras::ForwardDiffOneArgPushforwardExtras{T}
) where {F,T}
    ydual = compute_ydual_onearg(f, x, dx, extras)
    myderivative!(T, dy, ydual)
    return dy
end

## Derivative

function DI.prepare_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    return NoDerivativeExtras()
end

function DI.value_and_derivative(
    f::F, backend::AutoForwardDiff, x, ::NoDerivativeExtras
) where {F}
    T = tag_type(f, backend, x)
    ydual = f(make_dual(T, x, one(x)))
    return myvalue(T, ydual), myderivative(T, ydual)
end

function DI.value_and_derivative!(
    f::F, der, backend::AutoForwardDiff, x, ::NoDerivativeExtras
) where {F}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    ydual = f(xdual)
    return myvalue(T, ydual), myderivative!(T, der, ydual)
end

## Second derivative

function DI.prepare_second_derivative(f::F, backend::AutoForwardDiff, x) where {F}
    return NoSecondDerivativeExtras()
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
    f::F, backend::AutoForwardDiff, x, extras::ForwardDiffGradientExtras
) where {F}
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, backend, x, extras)
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

## Hessian

struct ForwardDiffHessianExtras{C1,C2} <: HessianExtras
    array_config::C1
    result_config::C2
end

function DI.prepare_hessian(f, backend::AutoForwardDiff, x)
    example_result = MutableDiffResult(
        one(eltype(x)), (similar(x), similar(x, length(x), length(x)))
    )
    chunk = choose_chunk(backend, x)
    array_config = HessianConfig(f, x, chunk)
    result_config = HessianConfig(f, example_result, x, chunk)
    return ForwardDiffHessianExtras(array_config, result_config)
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
    result = hessian!(result, f, x, extras.result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f::F, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras
) where {F}
    result = MutableDiffResult(
        one(eltype(x)), (similar(x), similar(x, length(x), length(x)))
    )
    result = hessian!(result, f, x, extras.result_config)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

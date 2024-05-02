## Pushforward

struct ForwardDiffOneArgPushforwardExtras{T,X} <: PushforwardExtras
    xdual_tmp::X
end

function DI.prepare_pushforward(f::F, backend::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, backend, x)
    xdual_tmp = make_dual(T, x, dx)
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
    @unpack xdual_tmp = extras
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

struct ForwardDiffHessianExtras{C} <: HessianExtras
    config::C
end

function DI.prepare_hessian(f, backend::AutoForwardDiff, x)
    return ForwardDiffHessianExtras(HessianConfig(f, x, choose_chunk(backend, x)))
end

function DI.hessian!(
    f::F, hess, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras
) where {F}
    return hessian!(hess, f, x, extras.config)
end

function DI.hessian(f::F, ::AutoForwardDiff, x, extras::ForwardDiffHessianExtras) where {F}
    return hessian(f, x, extras.config)
end

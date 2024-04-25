## Pushforward

DI.prepare_pushforward(f, ::AutoForwardDiff, x, dx) = NoPushforwardExtras()

function DI.value_and_pushforward(f, ::AutoForwardDiff, x, dx, ::NoPushforwardExtras)
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = myvalue(T, ydual)
    new_dy = myderivative(T, ydual)
    return y, new_dy
end

## Gradient

struct ForwardDiffGradientExtras{C} <: GradientExtras
    config::C
end

function DI.prepare_gradient(f, backend::AutoForwardDiff, x::AbstractArray)
    return ForwardDiffGradientExtras(GradientConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_gradient!(
    f,
    grad::AbstractArray,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffGradientExtras,
)
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f, backend::AutoForwardDiff, x::AbstractArray, extras::ForwardDiffGradientExtras
)
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, backend, x, extras)
end

function DI.gradient!(
    f,
    grad::AbstractArray,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffGradientExtras,
)
    return gradient!(grad, f, x, extras.config)
end

function DI.gradient(
    f, ::AutoForwardDiff, x::AbstractArray, extras::ForwardDiffGradientExtras
)
    return gradient(f, x, extras.config)
end

## Jacobian

struct ForwardDiffOneArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(f, backend::AutoForwardDiff, x::AbstractArray)
    return ForwardDiffOneArgJacobianExtras(JacobianConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffOneArgJacobianExtras,
)
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f, ::AutoForwardDiff, x::AbstractArray, extras::ForwardDiffOneArgJacobianExtras
)
    return f(x), jacobian(f, x, extras.config)
end

function DI.jacobian!(
    f,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffOneArgJacobianExtras,
)
    return jacobian!(jac, f, x, extras.config)
end

function DI.jacobian(
    f, ::AutoForwardDiff, x::AbstractArray, extras::ForwardDiffOneArgJacobianExtras
)
    return jacobian(f, x, extras.config)
end

## Hessian

struct ForwardDiffHessianExtras{C} <: HessianExtras
    config::C
end

function DI.prepare_hessian(f, backend::AutoForwardDiff, x::AbstractArray)
    return ForwardDiffHessianExtras(HessianConfig(f, x, choose_chunk(backend, x)))
end

function DI.hessian!(
    f,
    hess::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffHessianExtras,
)
    return hessian!(hess, f, x, extras.config)
end

function DI.hessian(
    f, ::AutoForwardDiff, x::AbstractArray, extras::ForwardDiffHessianExtras
)
    return hessian(f, x, extras.config)
end

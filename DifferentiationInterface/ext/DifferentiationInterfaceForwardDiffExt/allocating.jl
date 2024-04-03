## Pushforward

DI.prepare_pushforward(f, ::AnyAutoForwardDiff, x) = NoPushforwardExtras()

function DI.value_and_pushforward(f, ::AnyAutoForwardDiff, x, dx, ::NoPushforwardExtras)
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

function DI.prepare_gradient(f, backend::AnyAutoForwardDiff, x::AbstractArray)
    return ForwardDiffGradientExtras(GradientConfig(f, x, choose_chunk(backend, x)))
end

function DI.value_and_gradient!!(
    f,
    grad::AbstractArray,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffGradientExtras,
)
    result = DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f, backend::AnyAutoForwardDiff, x::AbstractArray, extras::ForwardDiffGradientExtras
)
    grad = similar(x)
    return DI.value_and_gradient!!(f, grad, backend, x, extras)
end

function DI.gradient!!(
    f,
    grad::AbstractArray,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffGradientExtras,
)
    return gradient!(grad, f, x, extras.config)
end

function DI.gradient(
    f, ::AnyAutoForwardDiff, x::AbstractArray, extras::ForwardDiffGradientExtras
)
    return gradient(f, x, extras.config)
end

## Jacobian

struct ForwardDiffAllocatingJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(f, backend::AnyAutoForwardDiff, x::AbstractArray)
    return ForwardDiffAllocatingJacobianExtras(
        JacobianConfig(f, x, choose_chunk(backend, x))
    )
end

function DI.value_and_jacobian!!(
    f,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffAllocatingJacobianExtras,
)
    y = f(x)
    result = DiffResult(y, jac)
    result = jacobian!(result, f, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f, ::AnyAutoForwardDiff, x::AbstractArray, extras::ForwardDiffAllocatingJacobianExtras
)
    return f(x), jacobian(f, x, extras.config)
end

function DI.jacobian!!(
    f,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffAllocatingJacobianExtras,
)
    return jacobian!(jac, f, x, extras.config)
end

function DI.jacobian(
    f, ::AnyAutoForwardDiff, x::AbstractArray, extras::ForwardDiffAllocatingJacobianExtras
)
    return jacobian(f, x, extras.config)
end

## Hessian

struct ForwardDiffHessianExtras{C} <: HessianExtras
    config::C
end

function DI.prepare_hessian(f, backend::AnyAutoForwardDiff, x::AbstractArray)
    return ForwardDiffHessianExtras(HessianConfig(f, x, choose_chunk(backend, x)))
end

function DI.hessian!!(
    f,
    hess::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffHessianExtras,
)
    return hessian!(hess, f, x, extras.config)
end

function DI.hessian(
    f, ::AnyAutoForwardDiff, x::AbstractArray, extras::ForwardDiffHessianExtras
)
    return hessian(f, x, extras.config)
end

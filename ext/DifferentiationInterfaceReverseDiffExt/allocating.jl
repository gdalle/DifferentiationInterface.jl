## Pullback

DI.prepare_pullback(f, ::AnyAutoReverseDiff, x) = NoPullbackExtras()

function DI.value_and_pullback!!(
    f,
    dx::AbstractArray,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    dy::Number,
    ::NoPullbackExtras,
)
    y = f(x)
    gradient!(dx, f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback(
    f, ::AnyAutoReverseDiff, x::AbstractArray, dy::Number, ::NoPullbackExtras
)
    y = f(x)
    dx = gradient(f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!!(
    f,
    dx::AbstractArray,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    ::NoPullbackExtras,
)
    y = f(x)
    jac = jacobian(f, x)  # allocates
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback(
    f, ::AnyAutoReverseDiff, x::AbstractArray, dy::AbstractArray, ::NoPullbackExtras
)
    y = f(x)
    jac = jacobian(f, x)  # allocates
    dx = reshape(transpose(jac) * vec(dy), size(x))
    return y, dx
end

### Trick for unsupported scalar input

function DI.value_and_pullback(
    f, backend::AnyAutoReverseDiff, x::Number, dy, ::NoPullbackExtras
)
    x_array = [x]
    f_array = f ∘ only
    new_extras = DI.prepare_pullback(f_array, backend, x_array)
    y, dx_array = DI.value_and_pullback(f_array, backend, x_array, dy, new_extras)
    return y, only(dx_array)
end

## Gradient

struct ReverseDiffGradientExtras{T} <: GradientExtras
    tape::T
end

function DI.prepare_gradient(f, backend::AnyAutoReverseDiff, x::AbstractArray)
    tape = GradientTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return ReverseDiffGradientExtras(tape)
end

function DI.value_and_gradient!!(
    _f,
    grad::AbstractArray,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffGradientExtras,
)
    result = DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, backend::AnyAutoReverseDiff, x::AbstractArray, extras::ReverseDiffGradientExtras
)
    grad = similar(x)
    return DI.value_and_gradient!!(f, grad, backend, x, extras)
end

function DI.gradient!!(
    _f,
    grad::AbstractArray,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffGradientExtras,
)
    return gradient!(grad, extras.tape, x)
end

function DI.gradient(
    _f, ::AnyAutoReverseDiff, x::AbstractArray, extras::ReverseDiffGradientExtras
)
    return gradient!(extras.tape, x)
end

## Jacobian

struct ReverseDiffAllocatingJacobianExtras{T} <: JacobianExtras
    tape::T
end

function DI.prepare_jacobian(f, backend::AnyAutoReverseDiff, x::AbstractArray)
    tape = JacobianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return ReverseDiffAllocatingJacobianExtras(tape)
end

function DI.value_and_jacobian!!(
    f,
    jac::AbstractMatrix,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffAllocatingJacobianExtras,
)
    y = f(x)
    result = DiffResult(y, jac)
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, ::AnyAutoReverseDiff, x::AbstractArray, extras::ReverseDiffAllocatingJacobianExtras
)
    return f(x), jacobian!(extras.tape, x)
end

function DI.jacobian!!(
    _f,
    jac::AbstractMatrix,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffAllocatingJacobianExtras,
)
    return jacobian!(jac, extras.tape, x)
end

function DI.jacobian(
    f, ::AnyAutoReverseDiff, x::AbstractArray, extras::ReverseDiffAllocatingJacobianExtras
)
    return jacobian!(extras.tape, x)
end

## Hessian

struct ReverseDiffHessianExtras{T} <: HessianExtras
    tape::T
end

function DI.prepare_hessian(f, backend::AnyAutoReverseDiff, x::AbstractArray)
    tape = HessianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return ReverseDiffHessianExtras(tape)
end

function DI.hessian!!(
    _f,
    hess::AbstractMatrix,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffHessianExtras,
)
    return hessian!(hess, extras.tape, x)
end

function DI.hessian(
    _f, ::AnyAutoReverseDiff, x::AbstractArray, extras::ReverseDiffHessianExtras
)
    return hessian!(extras.tape, x)
end

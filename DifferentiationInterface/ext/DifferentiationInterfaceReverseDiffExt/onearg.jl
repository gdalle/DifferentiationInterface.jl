## Pullback

DI.prepare_pullback(f, ::AutoReverseDiff, x, dy) = NoPullbackExtras()

function DI.value_and_pullback(
    f, ::AutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras
)
    y = f(x)
    dx = if y isa Number
        dy .* gradient(f, x)
    elseif y isa AbstractArray
        gradient(z -> dot(f(z), dy), x)
    end
    return y, dx
end

function DI.value_and_pullback!(
    f, dx, ::AutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras
)
    y = f(x)
    dx = if y isa Number
        dx = gradient!(dx, f, x)
        dx .*= dy
    elseif y isa AbstractArray
        gradient!(dx, z -> dot(f(z), dy), x)
    end
    return y, dx
end

function DI.value_and_pullback(
    f, backend::AutoReverseDiff, x::Number, dy, ::NoPullbackExtras
)
    x_array = [x]
    f_array = f ∘ only
    y, dx_array = DI.value_and_pullback(f_array, backend, x_array, dy)
    return y, only(dx_array)
end

## Gradient

struct ReverseDiffGradientExtras{T} <: GradientExtras
    tape::T
end

function DI.prepare_gradient(
    f, ::AutoReverseDiff{Compile}, x::AbstractArray
) where {Compile}
    tape = GradientTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffGradientExtras(tape)
end

function DI.value_and_gradient!(
    _f,
    grad::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffGradientExtras,
)
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    println(result)
    result = gradient!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, backend::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffGradientExtras
)
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, backend, x, extras)
end

function DI.gradient!(
    _f,
    grad::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffGradientExtras,
)
    return gradient!(grad, extras.tape, x)
end

function DI.gradient(
    _f, ::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffGradientExtras
)
    return gradient!(extras.tape, x)
end

## Jacobian

struct ReverseDiffOneArgJacobianExtras{T} <: JacobianExtras
    tape::T
end

function DI.prepare_jacobian(
    f, ::AutoReverseDiff{Compile}, x::AbstractArray
) where {Compile}
    tape = JacobianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffOneArgJacobianExtras(tape)
end

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffOneArgJacobianExtras,
)
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, ::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffOneArgJacobianExtras
)
    return f(x), jacobian!(extras.tape, x)
end

function DI.jacobian!(
    _f,
    jac::AbstractMatrix,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffOneArgJacobianExtras,
)
    return jacobian!(jac, extras.tape, x)
end

function DI.jacobian(
    f, ::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffOneArgJacobianExtras
)
    return jacobian!(extras.tape, x)
end

## Hessian

struct ReverseDiffHessianExtras{T} <: HessianExtras
    tape::T
end

function DI.prepare_hessian(f, ::AutoReverseDiff{Compile}, x::AbstractArray) where {Compile}
    tape = HessianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffHessianExtras(tape)
end

function DI.hessian!(
    _f,
    hess::AbstractMatrix,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffHessianExtras,
)
    return hessian!(hess, extras.tape, x)
end

function DI.hessian(
    _f, ::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffHessianExtras
)
    return hessian!(extras.tape, x)
end

function DI.value_gradient_and_hessian!(
    _f,
    grad,
    hess::AbstractMatrix,
    ::AutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffHessianExtras,
)
    result = MutableDiffResult(one(eltype(x)), (grad, hess))
    result = hessian!(result, extras.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    _f, ::AutoReverseDiff, x::AbstractArray, extras::ReverseDiffHessianExtras
)
    result = MutableDiffResult(
        one(eltype(x)), (similar(x), similar(x, length(x), length(x)))
    )
    result = hessian!(result, extras.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

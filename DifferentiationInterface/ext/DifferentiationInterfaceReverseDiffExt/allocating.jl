## Pullback

DI.prepare_pullback(f, ::AnyAutoReverseDiff, x) = NoPullbackExtras()

function DI.value_and_pullback_split(
    f, ::AnyAutoReverseDiff, x::AbstractArray, ::NoPullbackExtras
)
    y = f(x)
    pullbackfunc = if y isa Number
        dy -> dy .* gradient(f, x)
    elseif y isa AbstractArray
        dy -> gradient(z -> dot(f(z), dy), x)
    end
    return y, pullbackfunc
end

function DI.value_and_pullback!!_split(
    f, ::AnyAutoReverseDiff, x::AbstractArray, ::NoPullbackExtras
)
    y = f(x)
    pullbackfunc!! = if y isa Number
        (dx, dy) -> begin
            dx = gradient!(dx, f, x)
            dx .*= dy
        end
    elseif y isa AbstractArray
        (dx, dy) -> gradient!(dx, z -> dot(f(z), dy), x)
    end
    return y, pullbackfunc!!
end

function DI.value_and_pullback_split(
    f, backend::AnyAutoReverseDiff, x::Number, ::NoPullbackExtras
)
    x_array = [x]
    f_array = f ∘ only
    y, pullbackfunc = DI.value_and_pullback_split(f_array, backend, x_array)
    return y, only ∘ pullbackfunc
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

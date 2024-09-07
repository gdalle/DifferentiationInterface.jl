## Pullback

DI.prepare_pullback(f, ::AutoReverseDiff, x, ty::Tangents) = NoPullbackExtras()

function DI.value_and_pullback(
    f, ::NoPullbackExtras, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    y = f(x)
    tx = map(ty) do dy
        if y isa Number
            dy .* gradient(f, x)
        elseif y isa AbstractArray
            gradient(z -> dot(f(z), dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback!(
    f, ::NoPullbackExtras, tx::Tangents, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    y = f(x)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        if y isa Number
            dx = gradient!(dx, f, x)
            dx .*= dy
        elseif y isa AbstractArray
            gradient!(dx, z -> dot(f(z), dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback(
    f, ::NoPullbackExtras, backend::AutoReverseDiff, x::Number, ty::Tangents
)
    x_array = [x]
    f_array = f ∘ only
    y, tx_array = DI.value_and_pullback(f_array, backend, x_array, ty)
    return y, Tangents(only.(tx_array.d)...)
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
    f,
    grad::AbstractArray,
    extras::ReverseDiffGradientExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad,))
    result = gradient!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, extras::ReverseDiffGradientExtras, backend::AutoReverseDiff, x::AbstractArray
)
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, extras, backend, x)
end

function DI.gradient!(
    _f,
    grad::AbstractArray,
    extras::ReverseDiffGradientExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return gradient!(grad, extras.tape, x)
end

function DI.gradient(
    _f, extras::ReverseDiffGradientExtras, ::AutoReverseDiff, x::AbstractArray
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
    extras::ReverseDiffOneArgJacobianExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, extras::ReverseDiffOneArgJacobianExtras, ::AutoReverseDiff, x::AbstractArray
)
    return f(x), jacobian!(extras.tape, x)
end

function DI.jacobian!(
    _f,
    jac::AbstractMatrix,
    extras::ReverseDiffOneArgJacobianExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return jacobian!(jac, extras.tape, x)
end

function DI.jacobian(
    f, extras::ReverseDiffOneArgJacobianExtras, ::AutoReverseDiff, x::AbstractArray
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
    extras::ReverseDiffHessianExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return hessian!(hess, extras.tape, x)
end

function DI.hessian(
    _f, extras::ReverseDiffHessianExtras, ::AutoReverseDiff, x::AbstractArray
)
    return hessian!(extras.tape, x)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess::AbstractMatrix,
    extras::ReverseDiffHessianExtras,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad, hess))
    result = hessian!(result, extras.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f, extras::ReverseDiffHessianExtras, ::AutoReverseDiff, x::AbstractArray
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (similar(x), similar(x, length(x), length(x))))
    result = hessian!(result, extras.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

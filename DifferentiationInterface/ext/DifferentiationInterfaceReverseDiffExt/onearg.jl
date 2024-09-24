## Pullback

DI.prepare_pullback(f, ::AutoReverseDiff, x, ty::Tangents) = NoPullbackPrep()

function DI.value_and_pullback(
    f, ::NoPullbackPrep, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
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
    f, ::NoPullbackPrep, tx::Tangents, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
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
    f, ::NoPullbackPrep, backend::AutoReverseDiff, x::Number, ty::Tangents
)
    x_array = [x]
    f_array = f ∘ only
    y, tx_array = DI.value_and_pullback(f_array, backend, x_array, ty)
    return y, Tangents(only.(tx_array.d)...)
end

## Gradient

struct ReverseDiffGradientPrep{T} <: GradientPrep
    tape::T
end

function DI.prepare_gradient(
    f, ::AutoReverseDiff{Compile}, x::AbstractArray
) where {Compile}
    tape = GradientTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffGradientPrep(tape)
end

function DI.value_and_gradient!(
    f,
    grad::AbstractArray,
    prep::ReverseDiffGradientPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad,))
    result = gradient!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff, x::AbstractArray
)
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, prep, backend, x)
end

function DI.gradient!(
    _f,
    grad::AbstractArray,
    prep::ReverseDiffGradientPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return gradient!(grad, prep.tape, x)
end

function DI.gradient(_f, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x::AbstractArray)
    return gradient!(prep.tape, x)
end

## Jacobian

struct ReverseDiffOneArgJacobianPrep{T} <: JacobianPrep
    tape::T
end

function DI.prepare_jacobian(
    f, ::AutoReverseDiff{Compile}, x::AbstractArray
) where {Compile}
    tape = JacobianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffOneArgJacobianPrep(tape)
end

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x::AbstractArray
)
    return f(x), jacobian!(prep.tape, x)
end

function DI.jacobian!(
    _f,
    jac::AbstractMatrix,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return jacobian!(jac, prep.tape, x)
end

function DI.jacobian(
    f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x::AbstractArray
)
    return jacobian!(prep.tape, x)
end

## Hessian

struct ReverseDiffHessianPrep{T} <: HessianPrep
    tape::T
end

function DI.prepare_hessian(f, ::AutoReverseDiff{Compile}, x::AbstractArray) where {Compile}
    tape = HessianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffHessianPrep(tape)
end

function DI.hessian!(
    _f,
    hess::AbstractMatrix,
    prep::ReverseDiffHessianPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    return hessian!(hess, prep.tape, x)
end

function DI.hessian(_f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x::AbstractArray)
    return hessian!(prep.tape, x)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess::AbstractMatrix,
    prep::ReverseDiffHessianPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad, hess))
    result = hessian!(result, prep.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x::AbstractArray
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (similar(x), similar(x, length(x), length(x))))
    result = hessian!(result, prep.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

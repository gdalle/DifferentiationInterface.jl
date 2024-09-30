## Pullback

function DI.prepare_pullback(
    f, ::AutoReverseDiff, x, ty::NTuple, contexts::Vararg{Context,C}
) where {C}
    return NoPullbackPrep()
end

function DI.value_and_pullback(
    f,
    ::NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    dotclosure(z, dy) = dot(fc(z), dy)
    tx = map(ty) do dy
        if y isa Number
            dy .* gradient(fc, x)
        elseif y isa AbstractArray
            gradient(Fix2(dotclosure, dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback!(
    f,
    ::NoPullbackPrep,
    tx::NTuple,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    dotclosure(z, dy) = dot(fc(z), dy)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        if y isa Number
            dx = gradient!(dx, fc, x)
            dx .*= dy
        elseif y isa AbstractArray
            gradient!(dx, Fix2(dotclosure, dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    ::NoPullbackPrep,
    backend::AutoReverseDiff,
    x::Number,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    x_array = [x]
    f_array(x_array, args...) = f(only(x_array), args...)
    y, tx_array = DI.value_and_pullback(f_array, backend, x_array, ty, contexts...)
    return y, only.(tx_array)
end

## Gradient

### Without contexts

struct ReverseDiffGradientPrep{T} <: GradientPrep
    tape::T
end

function DI.prepare_gradient(f, ::AutoReverseDiff{Compile}, x) where {Compile}
    tape = GradientTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffGradientPrep(tape)
end

function DI.value_and_gradient!(
    f, grad, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad,))
    result = gradient!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff, x
)
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, prep, backend, x)
end

function DI.gradient!(_f, grad, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x)
    return gradient!(grad, prep.tape, x)
end

function DI.gradient(_f, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x)
    return gradient!(prep.tape, x)
end

### With contexts

function DI.prepare_gradient(f, ::AutoReverseDiff, x, contexts::Vararg{Context,C}) where {C}
    return NoGradientPrep()
end

function DI.value_and_gradient!(
    f, grad, ::NoGradientPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad,))
    result = gradient!(result, fc, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f, prep::NoGradientPrep, backend::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    grad = similar(x)
    return DI.value_and_gradient!(f, grad, prep, backend, x, contexts...)
end

function DI.gradient!(
    f, grad, ::NoGradientPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return gradient!(grad, fc, x)
end

function DI.gradient(
    f, ::NoGradientPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return gradient(fc, x)
end

## Jacobian

### Without contexts

struct ReverseDiffOneArgJacobianPrep{T} <: JacobianPrep
    tape::T
end

function DI.prepare_jacobian(f, ::AutoReverseDiff{Compile}, x) where {Compile}
    tape = JacobianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffOneArgJacobianPrep(tape)
end

function DI.value_and_jacobian!(
    f, jac, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x
)
    y = f(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x)
    return f(x), jacobian!(prep.tape, x)
end

function DI.jacobian!(_f, jac, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x)
    return jacobian!(jac, prep.tape, x)
end

function DI.jacobian(f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff, x)
    return jacobian!(prep.tape, x)
end

### With contexts

function DI.prepare_jacobian(f, ::AutoReverseDiff, x, contexts::Vararg{Context,C}) where {C}
    return NoJacobianPrep()
end

function DI.value_and_jacobian!(
    f, jac, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return fc(x), jacobian(fc, x)
end

function DI.jacobian!(
    f, jac, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return jacobian!(jac, fc, x)
end

function DI.jacobian(
    f, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return jacobian(fc, x)
end

## Hessian

### Without contexts

struct ReverseDiffHessianPrep{T} <: HessianPrep
    tape::T
end

function DI.prepare_hessian(f, ::AutoReverseDiff{Compile}, x) where {Compile}
    tape = HessianTape(f, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffHessianPrep(tape)
end

function DI.hessian!(_f, hess, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x)
    return hessian!(hess, prep.tape, x)
end

function DI.hessian(_f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x)
    return hessian!(prep.tape, x)
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad, hess))
    result = hessian!(result, prep.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x
)
    y = f(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (similar(x), similar(x, length(x), length(x))))
    result = hessian!(result, prep.tape, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

### With contexts

function DI.prepare_hessian(f, ::AutoReverseDiff, x, contexts::Vararg{Context,C}) where {C}
    return NoHessianPrep()
end

function DI.hessian!(
    f, hess, ::NoHessianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return hessian!(hess, fc, x)
end

function DI.hessian(
    f, ::NoHessianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return hessian(fc, x)
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, ::NoHessianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (grad, hess))
    result = hessian!(result, fc, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

function DI.value_gradient_and_hessian(
    f, ::NoHessianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    y = fc(x)  # TODO: remove once ReverseDiff#251 is fixed
    result = MutableDiffResult(y, (similar(x), similar(x, length(x), length(x))))
    result = hessian!(result, fc, x)
    return (
        DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    )
end

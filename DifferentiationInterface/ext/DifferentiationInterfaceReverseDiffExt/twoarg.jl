## Pullback

function DI.prepare_pullback(
    f!, y, ::AutoReverseDiff, x, ty::NTuple, contexts::Vararg{Context,C}
) where {C}
    return NoPullbackPrep()
end

### Array in

function DI.value_and_pullback(
    f!,
    y,
    ::NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc! = with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    tx = map(ty) do dy
        return gradient(Fix2(dotclosure, dy), x)
    end
    fc!(y, x)
    return y, tx
end

function DI.value_and_pullback!(
    f!,
    y,
    tx::NTuple,
    ::NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc! = with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        gradient!(dx, Fix2(dotclosure, dy), x)
    end
    fc!(y, x)
    return y, tx
end

function DI.pullback(
    f!,
    y,
    ::NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc! = with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    tx = map(ty) do dy
        return gradient(Fix2(dotclosure, dy), x)
    end
    return tx
end

function DI.pullback!(
    f!,
    y,
    tx::NTuple,
    ::NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc! = with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        gradient!(dx, Fix2(dotclosure, dy), x)
    end
    return tx
end

### Number in, not supported

function DI.value_and_pullback(
    f!,
    y,
    ::NoPullbackPrep,
    backend::AutoReverseDiff,
    x::Number,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    x_array = [x]
    function f!_array(_y::AbstractArray, _x_array, args...)
        return f!(_y, only(_x_array), args...)
    end
    y, tx_array = DI.value_and_pullback(f!_array, y, backend, x_array, ty, contexts...)
    return y, only.(tx_array)
end

## Jacobian

### Without contexts

struct ReverseDiffTwoArgJacobianPrep{T} <: JacobianPrep
    tape::T
end

function DI.prepare_jacobian(f!, y, ::AutoReverseDiff{Compile}, x) where {Compile}
    tape = JacobianTape(f!, y, x)
    if Compile
        tape = compile(tape)
    end
    return ReverseDiffTwoArgJacobianPrep(tape)
end

function DI.value_and_jacobian(
    _f!, y, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff, x
)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian!(
    _f!, y, jac, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff, x
)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, prep.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.jacobian(_f!, _y, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff, x)
    jac = jacobian!(prep.tape, x)
    return jac
end

function DI.jacobian!(
    _f!, _y, jac, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff, x
)
    jac = jacobian!(jac, prep.tape, x)
    return jac
end

### With contexts

function DI.prepare_jacobian(
    f!, y, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    return NoJacobianPrep()
end

function DI.value_and_jacobian(
    f!, y, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc! = with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian!(
    f!, y, jac, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc! = with_contexts(f!, contexts...)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.jacobian(
    f!, y, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc! = with_contexts(f!, contexts...)
    jac = jacobian(fc!, y, x)
    return jac
end

function DI.jacobian!(
    f!, y, jac, ::NoJacobianPrep, ::AutoReverseDiff, x, contexts::Vararg{Context,C}
) where {C}
    fc! = with_contexts(f!, contexts...)
    jac = jacobian!(jac, fc!, y, x)
    return jac
end

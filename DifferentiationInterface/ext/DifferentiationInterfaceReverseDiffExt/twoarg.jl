## Pullback

DI.prepare_pullback(f!, y, ::AutoReverseDiff, x, ty::Tangents) = NoPullbackPrep()

### Array in

function DI.value_and_pullback(
    f!, y, ::NoPullbackPrep, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    tx = map(ty) do dy
        function dotproduct_closure(x)
            y_copy = similar(y, eltype(x))
            f!(y_copy, x)
            return dot(y_copy, dy)
        end
        gradient(dotproduct_closure, x)
    end
    f!(y, x)
    return y, tx
end

function DI.value_and_pullback!(
    f!, y, tx::Tangents, ::NoPullbackPrep, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        function dotproduct_closure(x)
            y_copy = similar(y, eltype(x))
            f!(y_copy, x)
            return dot(y_copy, dy)
        end
        gradient!(dx, dotproduct_closure, x)
    end
    f!(y, x)
    return y, tx
end

function DI.pullback(
    f!, y, ::NoPullbackPrep, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    tx = map(ty) do dy
        function dotproduct_closure(x)
            y_copy = similar(y, eltype(x))
            f!(y_copy, x)
            return dot(y_copy, dy)
        end
        gradient(dotproduct_closure, x)
    end
    return tx
end

function DI.pullback!(
    f!, y, tx::Tangents, ::NoPullbackPrep, ::AutoReverseDiff, x::AbstractArray, ty::Tangents
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        function dotproduct_closure(x)
            y_copy = similar(y, eltype(x))
            f!(y_copy, x)
            return dot(y_copy, dy)
        end
        gradient!(dx, dotproduct_closure, x)
    end
    return tx
end

### Number in, not supported

function DI.value_and_pullback(
    f!, y, ::NoPullbackPrep, backend::AutoReverseDiff, x::Number, ty::Tangents{B}
) where {B}
    x_array = [x]
    f!_array(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    y, tx_array = DI.value_and_pullback(f!_array, y, backend, x_array, ty)
    return y, Tangents(only.(tx_array.d)...)
end

## Jacobian

struct ReverseDiffTwoArgJacobianPrep{T} <: JacobianPrep
    tape::T
end

function DI.prepare_jacobian(
    f!, y::AbstractArray, ::AutoReverseDiff{Compile}, x::AbstractArray
) where {Compile}
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

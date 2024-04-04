## Pullback

DI.prepare_pullback(f!, ::AnyAutoReverseDiff, y, x) = NoPullbackExtras()

function DI.value_and_pullback!!(
    f!,
    y::AbstractArray,
    dx::AbstractArray,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    ::NoPullbackExtras,
)
    jac = jacobian(f!, y, x)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

### Trick for unsupported scalar input

function DI.value_and_pullback!!(
    f!,
    y::AbstractArray,
    _dx::Number,
    backend::AnyAutoReverseDiff,
    x::Number,
    dy::AbstractArray,
    ::NoPullbackExtras,
)
    x_array = [x]
    dx_array = similar(x_array)
    f!_array(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    new_extras = DI.prepare_pullback(f!_array, backend, y, x_array)
    y, dx_array = DI.value_and_pullback!!(
        f!_array, y, dx_array, backend, x_array, dy, new_extras
    )
    return y, only(dx_array)
end

## Jacobian

struct ReverseDiffMutatingJacobianExtras{T} <: JacobianExtras
    tape::T
end

function DI.prepare_jacobian(
    f!, backend::AnyAutoReverseDiff, y::AbstractArray, x::AbstractArray
)
    tape = JacobianTape(f!, y, x)
    if backend.compile
        tape = compile(tape)
    end
    return ReverseDiffMutatingJacobianExtras(tape)
end

function DI.value_and_jacobian!!(
    _f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoReverseDiff,
    x::AbstractArray,
    extras::ReverseDiffMutatingJacobianExtras,
)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

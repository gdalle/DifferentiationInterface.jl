## Pullback

function DI.value_and_pullback!!(
    f!,
    y::AbstractArray,
    dx::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
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
    backend::AutoReverseDiff,
    x::Number,
    dy::AbstractArray,
    extras::Nothing,
)
    x_array = [x]
    dx_array = similar(x_array)
    f!_only(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    y, dx_array = DI.value_and_pullback!!(f!_only, y, dx_array, backend, x_array, dy)
    return y, only(dx_array)
end

## Jacobian

function DI.prepare_jacobian(
    f!, backend::AutoReverseDiff, y::AbstractArray, x::AbstractArray
)
    tape = JacobianTape(f!, y, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.value_and_jacobian!!(
    _f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoReverseDiff,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

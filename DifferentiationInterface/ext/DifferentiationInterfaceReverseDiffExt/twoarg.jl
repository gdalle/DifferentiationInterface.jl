## Pullback

DI.prepare_pullback(f!, ::AnyAutoReverseDiff, y, x) = NoPullbackExtras()

### Array in

function DI.value_and_pullback(
    f!, y, ::AnyAutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras
)
    function dotproduct_closure(x)
        y_copy = similar(y, eltype(x))
        f!(y_copy, x)
        return dot(y_copy, dy)
    end
    dx = gradient(dotproduct_closure, x)
    f!(y, x)
    return y, dx
end

function DI.value_and_pullback!(
    f!, y, dx, ::AnyAutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras
)
    function dotproduct_closure(x)
        y_copy = similar(y, eltype(x))
        f!(y_copy, x)
        return dot(y_copy, dy)
    end
    dx = gradient!(dx, dotproduct_closure, x)
    f!(y, x)
    return y, dx
end

function DI.pullback(f!, y, ::AnyAutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras)
    function dotproduct_closure(x)
        y_copy = similar(y, eltype(x))
        f!(y_copy, x)
        return dot(y_copy, dy)
    end
    dx = gradient(dotproduct_closure, x)
    return dx
end

function DI.pullback!(
    f!, y, dx, ::AnyAutoReverseDiff, x::AbstractArray, dy, ::NoPullbackExtras
)
    function dotproduct_closure(x)
        y_copy = similar(y, eltype(x))
        f!(y_copy, x)
        return dot(y_copy, dy)
    end
    dx = gradient!(dx, dotproduct_closure, x)
    return dx
end

### Number in, not supported

function DI.value_and_pullback(
    f!, y, backend::AnyAutoReverseDiff, x::Number, dy, ::NoPullbackExtras
)
    x_array = [x]
    dx_array = similar(x_array)
    f!_array(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    new_extras = DI.prepare_pullback(f!_array, backend, y, x_array)
    y, dx_array = DI.value_and_pullback(f!_array, y, backend, x_array, dy, new_extras)
    return y, only(dx_array)
end

## Jacobian

struct ReverseDiffTwoArgJacobianExtras{T} <: JacobianExtras
    tape::T
end

function DI.prepare_jacobian(
    f!, backend::AnyAutoReverseDiff, y::AbstractArray, x::AbstractArray
)
    tape = JacobianTape(f!, y, x)
    if backend.compile
        tape = compile(tape)
    end
    return ReverseDiffTwoArgJacobianExtras(tape)
end

function DI.value_and_jacobian(
    _f!, y, ::AnyAutoReverseDiff, x, extras::ReverseDiffTwoArgJacobianExtras
)
    jac = similar(y, length(y), length(x))
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian!(
    _f!, y, jac, ::AnyAutoReverseDiff, x, extras::ReverseDiffTwoArgJacobianExtras
)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, extras.tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.jacobian(
    _f!, _y, ::AnyAutoReverseDiff, x, extras::ReverseDiffTwoArgJacobianExtras
)
    jac = jacobian!(extras.tape, x)
    return jac
end

function DI.jacobian!(
    _f!, _y, jac, ::AnyAutoReverseDiff, x, extras::ReverseDiffTwoArgJacobianExtras
)
    jac = jacobian!(jac, extras.tape, x)
    return jac
end

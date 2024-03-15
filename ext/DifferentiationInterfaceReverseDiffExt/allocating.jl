## Pullback

function DI.value_and_pullback!(
    dx::AbstractArray, ::AutoReverseDiff, f, x::AbstractArray, dy::Real, extras::Nothing
)
    res = DiffResults.DiffResult(zero(dy), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    ::AutoReverseDiff,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
)
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    jac = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback!(
    _dx::Number, backend::AutoReverseDiff, f, x::Number, dy, extras::Nothing
)
    x_array = [x]
    dx_array = similar(x_array)
    y, dx_array = DI.value_and_pullback!(dx_array, backend, f ∘ only, x_array, dy, extras)
    return y, only(dx_array)
end

## Gradient

### Unprepared

function DI.value_and_gradient!(
    grad::AbstractArray, backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_gradient!(grad, backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.value_and_gradient(
    backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_gradient(backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.gradient!(
    grad::AbstractArray, backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.gradient!(grad, backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.gradient(backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing)
    return DI.gradient(backend, f, x, DI.prepare_gradient(backend, f, x))
end

### Prepared

function DI.value_and_gradient!(
    grad::AbstractArray,
    ::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    result = DiffResults.DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    grad = similar(x)
    return DI.value_and_gradient!(grad, backend, f, x, tape)
end

function DI.gradient!(
    grad::AbstractArray,
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    gradient!(grad, tape, x)
    return grad
end

function DI.gradient(
    ::AutoReverseDiff, f, x::AbstractArray, tape::Union{GradientTape,CompiledGradient}
)
    return gradient!(tape, x)
end

## Jacobian

### Unprepared

function DI.value_and_jacobian!(
    jac::AbstractArray, backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_jacobian!(jac, backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.value_and_jacobian(
    backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_jacobian(backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.jacobian!(
    jac::AbstractArray, backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.jacobian!(jac, backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.jacobian(backend::AutoReverseDiff, f, x::AbstractArray, extras::Nothing)
    return DI.jacobian(backend, f, x, DI.prepare_jacobian(backend, f, x))
end

### Prepared

function DI.value_and_jacobian!(
    jac::AbstractArray,
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    y = f(x)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    y = f(x)
    jac = jacobian!(tape, x)
    return y, jac
end

function DI.jacobian!(
    jac::AbstractArray,
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    jacobian!(jac, tape, x)
    return jac
end

function DI.jacobian(
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    return jacobian!(tape, x)
end

## Preparation

function DI.prepare_gradient(backend::AutoReverseDiff, f, x::AbstractArray)
    tape = GradientTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.prepare_jacobian(backend::AutoReverseDiff, f, x::AbstractArray)
    tape = JacobianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

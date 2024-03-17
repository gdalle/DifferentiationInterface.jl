## Gradient

function DI.prepare_gradient(backend::AutoReverseDiff, f, x::AbstractArray)
    tape = GradientTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

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

function DI.prepare_jacobian(backend::AutoReverseDiff, f, x::AbstractArray)
    tape = JacobianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix,
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
    jac::AbstractMatrix,
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

## Pullback

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    dy::Real,
    extras::Nothing,
)
    y, dx = DI.value_and_gradient!(dx, backend, f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoReverseDiff,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
)
    y, jac = DI.value_and_jacobian(backend, f, x)  # allocates
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback!(
    _dx::Number,
    backend::AutoReverseDiff,
    f,
    x::Number,
    dy::Union{Number,AbstractArray},
    extras::Nothing,
)
    x_array = [x]
    dx_array = similar(x_array)
    y, dx_array = DI.value_and_pullback!(dx_array, backend, f ∘ only, x_array, dy, extras)
    return y, only(dx_array)
end

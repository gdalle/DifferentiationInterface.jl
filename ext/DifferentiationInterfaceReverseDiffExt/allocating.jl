## Pullback

function DI.value_and_pullback!!(
    f,
    dx::AbstractArray,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    dy::Number,
    extras::Nothing,
)
    y = f(x)
    gradient!(dx, f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback(
    f, ::AllAutoReverseDiff, x::AbstractArray, dy::Number, extras::Nothing
)
    y = f(x)
    dx = gradient(f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!!(
    f,
    dx::AbstractArray,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
)
    y = f(x)
    jac = jacobian(f, x)  # allocates
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback(
    f, ::AllAutoReverseDiff, x::AbstractArray, dy::AbstractArray, extras::Nothing
)
    y = f(x)
    jac = jacobian(f, x)  # allocates
    dx = reshape(transpose(jac) * vec(dy), size(x))
    return y, dx
end

### Trick for unsupported scalar input

function DI.value_and_pullback(
    f, backend::AllAutoReverseDiff, x::Number, dy, extras::Nothing
)
    x_array = [x]
    y, dx_array = DI.value_and_pullback(f ∘ only, backend, x_array, dy)
    return y, only(dx_array)
end

## Gradient

function DI.prepare_gradient(f, backend::AllAutoReverseDiff, x::AbstractArray)
    tape = GradientTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.value_and_gradient!!(
    _f,
    grad::AbstractArray,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    result = DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_gradient(
    f,
    backend::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    grad = similar(x)
    return DI.value_and_gradient!!(f, grad, backend, x, tape)
end

function DI.gradient!!(
    _f,
    grad::AbstractArray,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{GradientTape,CompiledGradient},
)
    return gradient!(grad, tape, x)
end

function DI.gradient(
    _f, ::AllAutoReverseDiff, x::AbstractArray, tape::Union{GradientTape,CompiledGradient}
)
    return gradient!(tape, x)
end

## Jacobian

function DI.prepare_jacobian(f, backend::AllAutoReverseDiff, x::AbstractArray)
    tape = JacobianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.value_and_jacobian!!(
    f,
    jac::AbstractMatrix,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    y = f(x)
    result = DiffResult(y, jac)
    result = jacobian!(result, tape, x)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian(
    f, ::AllAutoReverseDiff, x::AbstractArray, tape::Union{JacobianTape,CompiledJacobian}
)
    return f(x), jacobian!(tape, x)
end

function DI.jacobian!!(
    _f,
    jac::AbstractMatrix,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{JacobianTape,CompiledJacobian},
)
    return jacobian!(jac, tape, x)
end

function DI.jacobian(
    f, ::AllAutoReverseDiff, x::AbstractArray, tape::Union{JacobianTape,CompiledJacobian}
)
    return jacobian!(tape, x)
end

## Hessian

function DI.prepare_hessian(f, backend::AllAutoReverseDiff, x::AbstractArray)
    tape = HessianTape(f, x)
    if backend.compile
        tape = compile(tape)
    end
    return tape
end

function DI.hessian!!(
    _f,
    hess::AbstractMatrix,
    ::AllAutoReverseDiff,
    x::AbstractArray,
    tape::Union{HessianTape,CompiledHessian},
)
    return hessian!(hess, tape, x)
end

function DI.hessian(
    _f, ::AllAutoReverseDiff, x::AbstractArray, tape::Union{HessianTape,CompiledHessian}
)
    return hessian!(tape, x)
end

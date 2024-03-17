
## Derivative

function DI.prepare_derivative(::AutoFastDifferentiation, f, x::Number)
    x_var = only(make_variables(:x))
    y_var = f(x_var)
    y_var_der = derivative(y_var, x_var)
    f_and_der_exe = make_function([y_var, y_var_der], [x_var])
    return f_and_der_exe
end

function DI.value_and_derivative(
    ::AutoFastDifferentiation, f, x::Number, f_and_der_exe::RuntimeGeneratedFunction
)
    y, der = f_and_der_exe(x)
    return y, der
end

function DI.derivative(
    ::AutoFastDifferentiation, f, x::Number, f_and_der_exe::RuntimeGeneratedFunction
)
    _, der = f_and_der_exe(x)
    return der
end

## Multiderivative

function DI.prepare_multiderivative(::AutoFastDifferentiation, f, x::Number)
    x_var = only(make_variables(:x))
    y_var = f(x_var)
    y_var_der = derivative(y_var, x_var)
    der_exe! = make_function(y_var_der, [x_var]; in_place=true)
    return der_exe!
end

function DI.value_and_multiderivative!(
    multider::AbstractArray,
    ::AutoFastDifferentiation,
    f,
    x::Number,
    der_exe!::RuntimeGeneratedFunction,
)
    y = f(x)
    der_exe!(multider, x)
    return y, multider
end

function DI.multiderivative!(
    multider::AbstractArray,
    ::AutoFastDifferentiation,
    f,
    x::Number,
    der_exe!::RuntimeGeneratedFunction,
)
    der_exe!(multider, x)
    return multider
end

## Gradient

function DI.prepare_gradient(::AutoFastDifferentiation, f, x::AbstractArray)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)
    grad_var = jacobian([y_var], x_var)[1, :]
    grad_exe! = make_function(grad_var, x_var; in_place=true)
    return grad_exe!
end

function DI.value_and_gradient!(
    grad::AbstractArray,
    ::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    grad_exe!::RuntimeGeneratedFunction,
)
    y = f(x)
    grad_exe!(grad, x)
    return y, grad
end

function DI.gradient!(
    grad::AbstractArray,
    ::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    grad_exe!::RuntimeGeneratedFunction,
)
    grad_exe!(grad, x)
    return grad
end

## Jacobian

function DI.prepare_jacobian(::AutoFastDifferentiation, f, x::AbstractArray)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)
    jac_var = jacobian(y_var, x_var)
    jac_exe! = make_function(jac_var, x_var; in_place=true)
    return jac_exe!
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix,
    ::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    jac_exe!::RuntimeGeneratedFunction,
)
    y = f(x)
    jac_exe!(jac, x)
    return y, jac
end

function DI.jacobian!(
    jac::AbstractMatrix,
    ::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    jac_exe!::RuntimeGeneratedFunction,
)
    jac_exe!(jac, x)
    return jac
end

## Pushforward

### Preparation

function DI.prepare_pushforward(backend::AutoFastDifferentiation, f, x)
    y = f(x)
    return prepare_pushforward_aux(backend, f, x, y)
end

function prepare_pushforward_aux(backend::AutoFastDifferentiation, f, x::Number, y::Number)
    return DI.prepare_derivative(backend, f, x)
end

function prepare_pushforward_aux(
    backend::AutoFastDifferentiation, f, x::AbstractArray, y::Number
)
    return DI.prepare_gradient(backend, f, x)
end

function prepare_pushforward_aux(
    backend::AutoFastDifferentiation, f, x::Number, y::AbstractArray
)
    return DI.prepare_multiderivative(backend, f, x)
end

function prepare_pushforward_aux(
    backend::AutoFastDifferentiation, f, x::AbstractArray, y::AbstractArray
)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)
    jv_var, v_var = jacobian_times_v(y_var, x_var)
    jvp_exe! = make_function(jv_var, [x_var; v_var]; in_place=true)
    return jvp_exe!
end

### Execution

function DI.value_and_pushforward!(
    dy::Number,
    backend::AutoFastDifferentiation,
    f,
    x::Number,
    dx,
    exe::RuntimeGeneratedFunction,
)
    y, der = DI.value_and_derivative(backend, f, x, exe)
    return y, der * dx
end

function DI.value_and_pushforward!(
    dy::AbstractArray,
    backend::AutoFastDifferentiation,
    f,
    x::Number,
    dx,
    exe::RuntimeGeneratedFunction,
)
    y, dy = DI.value_and_multiderivative!(dy, backend, f, x, exe)
    dy .*= dx
    return y, dy
end

function DI.value_and_pushforward!(
    _dy::Number,
    backend::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    dx,
    exe::RuntimeGeneratedFunction,
)
    grad = similar(x)
    y, grad = DI.value_and_gradient!(grad, backend, f, x, exe)
    return y, dot(grad, dx)
end

function DI.value_and_pushforward!(
    dy::AbstractArray,
    backend::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    dx,
    jvp_exe!::RuntimeGeneratedFunction,
)
    y = f(x)
    jvp_exe!(vec(dy), vcat(vec(x), vec(dx)))
    return y, dy
end

## Pullback

### Preparation

function DI.prepare_pullback(backend::AutoFastDifferentiation, f, x)
    y = f(x)
    return prepare_pullback_aux(backend, f, x, y)
end

function prepare_pullback_aux(backend::AutoFastDifferentiation, f, x::Number, y::Number)
    return DI.prepare_derivative(backend, f, x)
end

function prepare_pullback_aux(
    backend::AutoFastDifferentiation, f, x::AbstractArray, y::Number
)
    return DI.prepare_gradient(backend, f, x)
end

function prepare_pullback_aux(
    backend::AutoFastDifferentiation, f, x::Number, y::AbstractArray
)
    return DI.prepare_multiderivative(backend, f, x)
end

function prepare_pullback_aux(
    backend::AutoFastDifferentiation, f, x::AbstractArray, y::AbstractArray
)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)
    vj_var, v_var = jacobian_transpose_v(y_var, x_var)
    vjp_exe! = make_function(vj_var, [x_var; v_var]; in_place=true)
    return vjp_exe!
end

### Execution

function DI.value_and_pullback!(
    _dx::Number,
    backend::AutoFastDifferentiation,
    f,
    x::Number,
    dy::Number,
    exe::RuntimeGeneratedFunction,
)
    y, der = DI.value_and_derivative(backend, f, x, exe)
    return y, der * dy
end

function DI.value_and_pullback!(
    _dx::Number,
    backend::AutoFastDifferentiation,
    f,
    x::Number,
    dy::AbstractArray,
    exe::RuntimeGeneratedFunction,
)
    y, multider = DI.value_and_multiderivative(backend, f, x, exe)
    return y, dot(multider, dy)
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    dy::Number,
    exe::RuntimeGeneratedFunction,
)
    y, dx = DI.value_and_gradient!(dx, backend, f, x, exe)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoFastDifferentiation,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    vjp_exe!::RuntimeGeneratedFunction,
)
    y = f(x)
    vjp_exe!(vec(dx), vcat(vec(x), vec(dy)))
    return y, dx
end

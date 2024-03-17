module DifferentiationInterfaceFastDifferentiationExt

using DifferentiationInterface: AutoFastDifferentiation
import DifferentiationInterface as DI
using DocStringExtensions
using FastDifferentiation: derivative, jacobian, make_function, make_variables
using RuntimeGeneratedFunctions: RuntimeGeneratedFunction

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

end

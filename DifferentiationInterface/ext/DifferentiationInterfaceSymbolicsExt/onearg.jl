## Pushforward

struct SymbolicsOneArgPushforwardExtras{E1,E2} <: PushforwardExtras
    pf_exe::E1
    pf_exe!::E2
end

function DI.prepare_pushforward(f, ::AutoSymbolics, x, dx)
    x_var = if x isa Number
        variable(:x)
    else
        variables(:x, axes(x)...)
    end
    dx_var = if dx isa Number
        variable(:dx)
    else
        variables(:dx, axes(dx)...)
    end
    t_var = variable(:t)
    step_der_var = derivative(f(x_var + t_var * dx_var), t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, vcat(myvec(x_var), myvec(dx_var)); expression=Val(false))
    (pf_exe, pf_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgPushforwardExtras(pf_exe, pf_exe!)
end

function DI.pushforward(f, ::AutoSymbolics, x, dx, extras::SymbolicsOneArgPushforwardExtras)
    v_vec = vcat(myvec(x), myvec(dx))
    dy = extras.pf_exe(v_vec)
    return dy
end

function DI.pushforward!(
    f, dy, ::AutoSymbolics, x, dx, extras::SymbolicsOneArgPushforwardExtras
)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.pf_exe!(dy, v_vec)
    return dy
end

function DI.value_and_pushforward(
    f, backend::AutoSymbolics, x, dx, extras::SymbolicsOneArgPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

function DI.value_and_pushforward!(
    f, dy, backend::AutoSymbolics, x, dx, extras::SymbolicsOneArgPushforwardExtras
)
    return f(x), DI.pushforward!(f, dy, backend, x, dx, extras)
end

## Derivative

struct SymbolicsOneArgDerivativeExtras{E1,E2} <: DerivativeExtras
    der_exe::E1
    der_exe!::E2
end

function DI.prepare_derivative(f, ::AutoSymbolics, x)
    x_var = variable(:x)
    der_var = derivative(f(x_var), x_var)

    res = build_function(der_var, x_var; expression=Val(false))
    (der_exe, der_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgDerivativeExtras(der_exe, der_exe!)
end

function DI.derivative(f, ::AutoSymbolics, x, extras::SymbolicsOneArgDerivativeExtras)
    return extras.der_exe(x)
end

function DI.derivative!(f, der, ::AutoSymbolics, x, extras::SymbolicsOneArgDerivativeExtras)
    extras.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f, backend::AutoSymbolics, x, extras::SymbolicsOneArgDerivativeExtras
)
    return f(x), DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative!(
    f, der, backend::AutoSymbolics, x, extras::SymbolicsOneArgDerivativeExtras
)
    return f(x), DI.derivative!(f, der, backend, x, extras)
end

## Gradient

struct SymbolicsOneArgGradientExtras{E1,E2} <: GradientExtras
    grad_exe::E1
    grad_exe!::E2
end

function DI.prepare_gradient(f, ::AutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.gradient only accepts vectors
    grad_var = gradient(f(x_var), vec(x_var))

    res = build_function(grad_var, vec(x_var); expression=Val(false))
    (grad_exe, grad_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgGradientExtras(grad_exe, grad_exe!)
end

function DI.gradient(f, ::AutoSymbolics, x, extras::SymbolicsOneArgGradientExtras)
    return reshape(extras.grad_exe(vec(x)), size(x))
end

function DI.gradient!(f, grad, ::AutoSymbolics, x, extras::SymbolicsOneArgGradientExtras)
    extras.grad_exe!(vec(grad), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, backend::AutoSymbolics, x, extras::SymbolicsOneArgGradientExtras
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.value_and_gradient!(
    f, grad, backend::AutoSymbolics, x, extras::SymbolicsOneArgGradientExtras
)
    return f(x), DI.gradient!(f, grad, backend, x, extras)
end

## Jacobian

struct SymbolicsOneArgJacobianExtras{E1,E2} <: JacobianExtras
    jac_exe::E1
    jac_exe!::E2
end

function DI.prepare_jacobian(f, backend::AnyAutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    jac_var = if issparse(backend)
        sparsejacobian(f(x_var), x_var)
    else
        jacobian(f(x_var), x_var)
    end

    res = build_function(jac_var, x_var; expression=Val(false))
    (jac_exe, jac_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgJacobianExtras(jac_exe, jac_exe!)
end

function DI.jacobian(f, ::AnyAutoSymbolics, x, extras::SymbolicsOneArgJacobianExtras)
    return extras.jac_exe(x)
end

function DI.jacobian!(f, jac, ::AnyAutoSymbolics, x, extras::SymbolicsOneArgJacobianExtras)
    extras.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f, backend::AnyAutoSymbolics, x, extras::SymbolicsOneArgJacobianExtras
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!(
    f, jac, backend::AnyAutoSymbolics, x, extras::SymbolicsOneArgJacobianExtras
)
    return f(x), DI.jacobian!(f, jac, backend, x, extras)
end

## Hessian

struct SymbolicsOneArgHessianExtras{E1,E2} <: HessianExtras
    hess_exe::E1
    hess_exe!::E2
end

function DI.prepare_hessian(f, backend::AnyAutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.gradient only accepts vectors
    hess_var = if issparse(backend)
        sparsehessian(f(x_var), vec(x_var))
    else
        hessian(f(x_var), vec(x_var))
    end

    res = build_function(hess_var, vec(x_var); expression=Val(false))
    (hess_exe, hess_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgHessianExtras(hess_exe, hess_exe!)
end

function DI.hessian(f, ::AnyAutoSymbolics, x, extras::SymbolicsOneArgHessianExtras)
    return extras.hess_exe(vec(x))
end

function DI.hessian!(f, hess, ::AnyAutoSymbolics, x, extras::SymbolicsOneArgHessianExtras)
    extras.hess_exe!(hess, vec(x))
    return hess
end

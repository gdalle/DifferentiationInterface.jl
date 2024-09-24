## Pushforward

struct SymbolicsOneArgPushforwardPrep{E1,E1!} <: PushforwardPrep
    pf_exe::E1
    pf_exe!::E1!
end

function DI.prepare_pushforward(f, ::AutoSymbolics, x, tx::Tangents)
    dx = first(tx)
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
    return SymbolicsOneArgPushforwardPrep(pf_exe, pf_exe!)
end

function DI.pushforward(
    f, prep::SymbolicsOneArgPushforwardPrep, ::AutoSymbolics, x, tx::Tangents
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        dy = prep.pf_exe(v_vec)
    end
    return ty
end

function DI.pushforward!(
    f, ty::Tangents, prep::SymbolicsOneArgPushforwardPrep, ::AutoSymbolics, x, tx::Tangents
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        v_vec = vcat(myvec(x), myvec(dx))
        prep.pf_exe!(dy, v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f, prep::SymbolicsOneArgPushforwardPrep, backend::AutoSymbolics, x, tx::Tangents
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::Tangents,
)
    return f(x), DI.pushforward!(f, ty, prep, backend, x, tx)
end

## Derivative

struct SymbolicsOneArgDerivativePrep{E1,E1!} <: DerivativePrep
    der_exe::E1
    der_exe!::E1!
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
    return SymbolicsOneArgDerivativePrep(der_exe, der_exe!)
end

function DI.derivative(f, prep::SymbolicsOneArgDerivativePrep, ::AutoSymbolics, x)
    return prep.der_exe(x)
end

function DI.derivative!(f, der, prep::SymbolicsOneArgDerivativePrep, ::AutoSymbolics, x)
    prep.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f, prep::SymbolicsOneArgDerivativePrep, backend::AutoSymbolics, x
)
    return f(x), DI.derivative(f, prep, backend, x)
end

function DI.value_and_derivative!(
    f, der, prep::SymbolicsOneArgDerivativePrep, backend::AutoSymbolics, x
)
    return f(x), DI.derivative!(f, der, prep, backend, x)
end

## Gradient

struct SymbolicsOneArgGradientPrep{E1,E1!} <: GradientPrep
    grad_exe::E1
    grad_exe!::E1!
end

function DI.prepare_gradient(f, ::AutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.gradient only accepts vectors
    grad_var = gradient(f(x_var), vec(x_var))

    res = build_function(grad_var, vec(x_var); expression=Val(false))
    (grad_exe, grad_exe!) = res
    return SymbolicsOneArgGradientPrep(grad_exe, grad_exe!)
end

function DI.gradient(f, prep::SymbolicsOneArgGradientPrep, ::AutoSymbolics, x)
    return reshape(prep.grad_exe(vec(x)), size(x))
end

function DI.gradient!(f, grad, prep::SymbolicsOneArgGradientPrep, ::AutoSymbolics, x)
    prep.grad_exe!(vec(grad), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, prep::SymbolicsOneArgGradientPrep, backend::AutoSymbolics, x
)
    return f(x), DI.gradient(f, prep, backend, x)
end

function DI.value_and_gradient!(
    f, grad, prep::SymbolicsOneArgGradientPrep, backend::AutoSymbolics, x
)
    return f(x), DI.gradient!(f, grad, prep, backend, x)
end

## Jacobian

struct SymbolicsOneArgJacobianPrep{E1,E1!} <: JacobianPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x
)
    x_var = variables(:x, axes(x)...)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(f(x_var)), vec(x_var))
    else
        jacobian(f(x_var), x_var)
    end

    res = build_function(jac_var, x_var; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsOneArgJacobianPrep(jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return prep.jac_exe(x)
end

function DI.jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    prep.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian(f, prep, backend, x)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian!(f, jac, prep, backend, x)
end

## Hessian

struct SymbolicsOneArgHessianPrep{G,E2,E2!} <: HessianPrep
    gradient_prep::G
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian(f, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.hessian only accepts vectors
    hess_var = if backend isa AutoSparse
        sparsehessian(f(x_var), vec(x_var))
    else
        hessian(f(x_var), vec(x_var))
    end

    res = build_function(hess_var, vec(x_var); expression=Val(false))
    (hess_exe, hess_exe!) = res

    gradient_prep = DI.prepare_gradient(f, dense_ad(backend), x)
    return SymbolicsOneArgHessianPrep(gradient_prep, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return prep.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    prep::SymbolicsOneArgHessianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    prep.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, grad = DI.value_and_gradient(f, prep.gradient_prep, dense_ad(backend), x)
    hess = DI.hessian(f, prep, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, _ = DI.value_and_gradient!(f, grad, prep.gradient_prep, dense_ad(backend), x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end

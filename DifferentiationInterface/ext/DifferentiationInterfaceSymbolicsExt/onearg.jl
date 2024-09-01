## Pushforward

struct SymbolicsOneArgPushforwardExtras{E1,E1!} <: PushforwardExtras
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
    return SymbolicsOneArgPushforwardExtras(pf_exe, pf_exe!)
end

function DI.pushforward(
    f, ::AutoSymbolics, x, tx::Tangents, extras::SymbolicsOneArgPushforwardExtras
)
    dys = map(tx.d) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        dy = extras.pf_exe(v_vec)
    end
    return Tangents(dys)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    extras::SymbolicsOneArgPushforwardExtras,
    ::AutoSymbolics,
    x,
    tx::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        v_vec = vcat(myvec(x), myvec(dx))
        extras.pf_exe!(dy, v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f, extras::SymbolicsOneArgPushforwardExtras, backend::AutoSymbolics, x, tx::Tangents
)
    return f(x), DI.pushforward(f, extras, backend, x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    extras::SymbolicsOneArgPushforwardExtras,
    backend::AutoSymbolics,
    x,
    tx::Tangents,
)
    return f(x), DI.pushforward!(f, ty, extras, backend, x, tx)
end

## Derivative

struct SymbolicsOneArgDerivativeExtras{E1,E1!} <: DerivativeExtras
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
    return SymbolicsOneArgDerivativeExtras(der_exe, der_exe!)
end

function DI.derivative(f, extras::SymbolicsOneArgDerivativeExtras, ::AutoSymbolics, x)
    return extras.der_exe(x)
end

function DI.derivative!(f, der, extras::SymbolicsOneArgDerivativeExtras, ::AutoSymbolics, x)
    extras.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f, extras::SymbolicsOneArgDerivativeExtras, backend::AutoSymbolics, x
)
    return f(x), DI.derivative(f, extras, backend, x)
end

function DI.value_and_derivative!(
    f, der, extras::SymbolicsOneArgDerivativeExtras, backend::AutoSymbolics, x
)
    return f(x), DI.derivative!(f, der, extras, backend, x)
end

## Gradient

struct SymbolicsOneArgGradientExtras{E1,E1!} <: GradientExtras
    grad_exe::E1
    grad_exe!::E1!
end

function DI.prepare_gradient(f, ::AutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.gradient only accepts vectors
    grad_var = gradient(f(x_var), vec(x_var))

    res = build_function(grad_var, vec(x_var); expression=Val(false))
    (grad_exe, grad_exe!) = res
    return SymbolicsOneArgGradientExtras(grad_exe, grad_exe!)
end

function DI.gradient(f, extras::SymbolicsOneArgGradientExtras, ::AutoSymbolics, x)
    return reshape(extras.grad_exe(vec(x)), size(x))
end

function DI.gradient!(f, grad, extras::SymbolicsOneArgGradientExtras, ::AutoSymbolics, x)
    extras.grad_exe!(vec(grad), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, extras::SymbolicsOneArgGradientExtras, backend::AutoSymbolics, x
)
    return f(x), DI.gradient(f, extras, backend, x)
end

function DI.value_and_gradient!(
    f, grad, extras::SymbolicsOneArgGradientExtras, backend::AutoSymbolics, x
)
    return f(x), DI.gradient!(f, grad, extras, backend, x)
end

## Jacobian

struct SymbolicsOneArgJacobianExtras{E1,E1!} <: JacobianExtras
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
    return SymbolicsOneArgJacobianExtras(jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    extras::SymbolicsOneArgJacobianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return extras.jac_exe(x)
end

function DI.jacobian!(
    f,
    jac,
    extras::SymbolicsOneArgJacobianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    extras.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f,
    extras::SymbolicsOneArgJacobianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian(f, extras, backend, x)
end

function DI.value_and_jacobian!(
    f,
    jac,
    extras::SymbolicsOneArgJacobianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian!(f, jac, extras, backend, x)
end

## Hessian

struct SymbolicsOneArgHessianExtras{G,E2,E2!} <: HessianExtras
    gradient_extras::G
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

    gradient_extras = DI.prepare_gradient(f, maybe_dense_ad(backend), x)
    return SymbolicsOneArgHessianExtras(gradient_extras, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    extras::SymbolicsOneArgHessianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return extras.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    extras::SymbolicsOneArgHessianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    extras.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    extras::SymbolicsOneArgHessianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, grad = DI.value_and_gradient(f, extras.gradient_extras, maybe_dense_ad(backend), x)
    hess = DI.hessian(f, extras, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    extras::SymbolicsOneArgHessianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, _ = DI.value_and_gradient!(
        f, grad, extras.gradient_extras, maybe_dense_ad(backend), x
    )
    DI.hessian!(f, hess, extras, backend, x)
    return y, grad, hess
end

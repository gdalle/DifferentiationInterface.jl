## Pushforward

struct SymbolicsTwoArgPushforwardExtras{E1,E1!} <: PushforwardExtras
    pushforward_exe::E1
    pushforward_exe!::E1!
end

function DI.prepare_pushforward(f!, y, ::AutoSymbolics, x, tx::Tangents)
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
    y_var = variables(:y, axes(y)...)
    t_var = variable(:t)
    f!(y_var, x_var + t_var * dx_var)
    step_der_var = derivative(y_var, t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, vcat(myvec(x_var), myvec(dx_var)); expression=Val(false))
    (pushforward_exe, pushforward_exe!) = res
    return SymbolicsTwoArgPushforwardExtras(pushforward_exe, pushforward_exe!)
end

function DI.pushforward(
    f!, y, extras::SymbolicsTwoArgPushforwardExtras, ::AutoSymbolics, x, tx::Tangents
)
    dys = map(tx.d) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        dy = extras.pushforward_exe(v_vec)
    end
    return Tangents(dys)
end

function DI.pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::SymbolicsTwoArgPushforwardExtras,
    ::AutoSymbolics,
    x,
    tx::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        v_vec = vcat(myvec(x), myvec(dx))
        extras.pushforward_exe!(dy, v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f!, y, extras::SymbolicsTwoArgPushforwardExtras, backend::AutoSymbolics, x, tx::Tangents
)
    ty = DI.pushforward(f!, y, extras, backend, x, tx)
    f!(y, x)
    return y, ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::SymbolicsTwoArgPushforwardExtras,
    backend::AutoSymbolics,
    x,
    tx::Tangents,
)
    DI.pushforward!(f!, y, ty, extras, backend, x, tx)
    f!(y, x)
    return y, ty
end

## Derivative

struct SymbolicsTwoArgDerivativeExtras{E1,E1!} <: DerivativeExtras
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f!, y, ::AutoSymbolics, x)
    x_var = variable(:x)
    y_var = variables(:y, axes(y)...)
    f!(y_var, x_var)
    der_var = derivative(y_var, x_var)

    res = build_function(der_var, x_var; expression=Val(false))
    (der_exe, der_exe!) = res
    return SymbolicsTwoArgDerivativeExtras(der_exe, der_exe!)
end

function DI.derivative(f!, y, extras::SymbolicsTwoArgDerivativeExtras, ::AutoSymbolics, x)
    return extras.der_exe(x)
end

function DI.derivative!(
    f!, y, der, extras::SymbolicsTwoArgDerivativeExtras, ::AutoSymbolics, x
)
    extras.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f!, y, extras::SymbolicsTwoArgDerivativeExtras, backend::AutoSymbolics, x
)
    der = DI.derivative(f!, y, extras, backend, x)
    f!(y, x)
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, extras::SymbolicsTwoArgDerivativeExtras, backend::AutoSymbolics, x
)
    DI.derivative!(f!, y, der, extras, x, backend)
    f!(y, x)
    return y, der
end

## Jacobian

struct SymbolicsTwoArgJacobianExtras{E1,E1!} <: JacobianExtras
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f!, y, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x
)
    x_var = variables(:x, axes(x)...)
    y_var = variables(:y, axes(y)...)
    f!(y_var, x_var)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(y_var), vec(x_var))
    else
        jacobian(y_var, x_var)
    end

    res = build_function(jac_var, x_var; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsTwoArgJacobianExtras(jac_exe, jac_exe!)
end

function DI.jacobian(
    f!,
    y,
    extras::SymbolicsTwoArgJacobianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return extras.jac_exe(x)
end

function DI.jacobian!(
    f!,
    y,
    jac,
    extras::SymbolicsTwoArgJacobianExtras,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    extras.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f!,
    y,
    extras::SymbolicsTwoArgJacobianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    jac = DI.jacobian(f!, y, extras, backend, x)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    extras::SymbolicsTwoArgJacobianExtras,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    DI.jacobian!(f!, y, jac, extras, x, backend)
    f!(y, x)
    return y, jac
end

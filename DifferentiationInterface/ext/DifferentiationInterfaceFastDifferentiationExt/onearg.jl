## Pushforward

struct FastDifferentiationOneArgPushforwardExtras{Y,E1,E1!} <: PushforwardExtras
    y_prototype::Y
    jvp_exe::E1
    jvp_exe!::E1!
end

function DI.prepare_pushforward(f, ::AutoFastDifferentiation, x, tx::Tangents)
    y_prototype = f(x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    jvp_exe! = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationOneArgPushforwardExtras(y_prototype, jvp_exe, jvp_exe!)
end

function DI.pushforward(
    f,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    dy = map(tx.d) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        if extras.y_prototype isa Number
            return only(extras.jvp_exe(v_vec))
        else
            return reshape(extras.jvp_exe(v_vec), size(extras.y_prototype))
        end
    end
    return Tangents(dy...)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        v_vec = vcat(myvec(x), myvec(dx))
        extras.jvp_exe!(vec(dy), v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    backend::AutoFastDifferentiation,
    x,
    tx::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    return f(x), DI.pushforward(f, backend, x, tx, extras)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    backend::AutoFastDifferentiation,
    x,
    tx::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    return f(x), DI.pushforward!(f, ty, backend, x, tx, extras)
end

## Pullback

struct FastDifferentiationOneArgPullbackExtras{E1,E1!} <: PullbackExtras
    vjp_exe::E1
    vjp_exe!::E1!
end

function DI.prepare_pullback(f, ::AutoFastDifferentiation, x, ty::Tangents)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    vjp_exe! = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationOneArgPullbackExtras(vjp_exe, vjp_exe!)
end

function DI.pullback(
    f,
    ::AutoFastDifferentiation,
    x,
    ty::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
)
    dx = map(ty.d) do dy
        v_vec = vcat(myvec(x), myvec(dy))
        if x isa Number
            return only(extras.vjp_exe(v_vec))
        else
            return reshape(extras.vjp_exe(v_vec), size(x))
        end
    end
    return Tangents(dx...)
end

function DI.pullback!(
    f,
    tx::Tangents,
    ::AutoFastDifferentiation,
    x,
    ty::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        v_vec = vcat(myvec(x), myvec(dy))
        extras.vjp_exe!(vec(dx), v_vec)
    end
    return tx
end

function DI.value_and_pullback(
    f,
    backend::AutoFastDifferentiation,
    x,
    ty::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
)
    return f(x), DI.pullback(f, backend, x, ty, extras)
end

function DI.value_and_pullback!(
    f,
    tx::Tangents,
    backend::AutoFastDifferentiation,
    x,
    ty::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
)
    return f(x), DI.pullback!(f, tx, backend, x, ty, extras)
end

## Derivative

struct FastDifferentiationOneArgDerivativeExtras{Y,E1,E1!} <: DerivativeExtras
    y_prototype::Y
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f, ::AutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgDerivativeExtras(y_prototype, der_exe, der_exe!)
end

function DI.derivative(
    f, ::AutoFastDifferentiation, x, extras::FastDifferentiationOneArgDerivativeExtras
)
    if extras.y_prototype isa Number
        return only(extras.der_exe(monovec(x)))
    else
        return reshape(extras.der_exe(monovec(x)), size(extras.y_prototype))
    end
end

function DI.derivative!(
    f, der, ::AutoFastDifferentiation, x, extras::FastDifferentiationOneArgDerivativeExtras
)
    extras.der_exe!(vec(der), monovec(x))
    return der
end

function DI.value_and_derivative(
    f,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgDerivativeExtras,
)
    return f(x), DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative!(
    f,
    der,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgDerivativeExtras,
)
    return f(x), DI.derivative!(f, der, backend, x, extras)
end

## Gradient

struct FastDifferentiationOneArgGradientExtras{E1,E1!} <: GradientExtras
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_gradient(f, backend::AutoFastDifferentiation, x)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    y_vec_var = monovec(y_var)
    jac_var = jacobian(y_vec_var, x_vec_var)
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgGradientExtras(jac_exe, jac_exe!)
end

function DI.gradient(
    f, ::AutoFastDifferentiation, x, extras::FastDifferentiationOneArgGradientExtras
)
    jac = extras.jac_exe(vec(x))
    grad_vec = @view jac[1, :]
    return reshape(grad_vec, size(x))
end

function DI.gradient!(
    f, grad, ::AutoFastDifferentiation, x, extras::FastDifferentiationOneArgGradientExtras
)
    extras.jac_exe!(reshape(grad, 1, length(grad)), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, backend::AutoFastDifferentiation, x, extras::FastDifferentiationOneArgGradientExtras
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.value_and_gradient!(
    f,
    grad,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgGradientExtras,
)
    return f(x), DI.gradient!(f, grad, backend, x, extras)
end

## Jacobian

struct FastDifferentiationOneArgJacobianExtras{Y,E1,E1!} <: JacobianExtras
    y_prototype::Y
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    y_prototype = f(x)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    jac_var = if backend isa AutoSparse
        sparse_jacobian(y_vec_var, x_vec_var)
    else
        jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgJacobianExtras(y_prototype, jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    return extras.jac_exe(vec(x))
end

function DI.jacobian!(
    f,
    jac,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    extras.jac_exe!(jac, vec(x))
    return jac
end

function DI.value_and_jacobian(
    f,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    return f(x), DI.jacobian!(f, jac, backend, x, extras)
end

## Second derivative

struct FastDifferentiationAllocatingSecondDerivativeExtras{Y,D,E2,E2!} <:
       SecondDerivativeExtras
    y_prototype::Y
    derivative_extras::D
    der2_exe::E2
    der2_exe!::E2!
end

function DI.prepare_second_derivative(f, backend::AutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)

    der2_vec_var = derivative(y_vec_var, x_var, x_var)
    der2_exe = make_function(der2_vec_var, x_vec_var; in_place=false)
    der2_exe! = make_function(der2_vec_var, x_vec_var; in_place=true)

    derivative_extras = DI.prepare_derivative(f, backend, x)
    return FastDifferentiationAllocatingSecondDerivativeExtras(
        y_prototype, derivative_extras, der2_exe, der2_exe!
    )
end

function DI.second_derivative(
    f,
    ::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    if extras.y_prototype isa Number
        return only(extras.der2_exe(monovec(x)))
    else
        return reshape(extras.der2_exe(monovec(x)), size(extras.y_prototype))
    end
end

function DI.second_derivative!(
    f,
    der2,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    extras.der2_exe!(vec(der2), monovec(x))
    return der2
end

function DI.value_derivative_and_second_derivative(
    f,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    y, der = DI.value_and_derivative(f, backend, x, extras.derivative_extras)
    der2 = DI.second_derivative(f, backend, x, extras)
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    backend::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    y, _ = DI.value_and_derivative!(f, der, backend, x, extras.derivative_extras)
    DI.second_derivative!(f, der2, backend, x, extras)
    return y, der, der2
end

## HVP

struct FastDifferentiationHVPExtras{E2,E2!} <: HVPExtras
    hvp_exe::E2
    hvp_exe!::E2!
end

function DI.prepare_hvp(f, ::AutoFastDifferentiation, x, tx::Tangents)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    hv_vec_var, v_vec_var = hessian_times_v(y_var, x_vec_var)
    hvp_exe = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    hvp_exe! = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationHVPExtras(hvp_exe, hvp_exe!)
end

function DI.hvp(
    f, ::AutoFastDifferentiation, x, tx::Tangents, extras::FastDifferentiationHVPExtras
)
    dg = map(tx.d) do dx
        v_vec = vcat(vec(x), vec(dx))
        dg_vec = extras.hvp_exe(v_vec)
        return reshape(dg_vec, size(x))
    end
    return Tangents(dg...)
end

function DI.hvp!(
    f,
    tg::Tangents,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
    extras::FastDifferentiationHVPExtras,
)
    for b in eachindex(tx.d, ty.d)
        dx, dg = tx.d[b], tg.d[b]
        v_vec = vcat(vec(x), vec(dx))
        extras.hvp_exe!(dg, v_vec)
    end
    return tg
end

## Hessian

struct FastDifferentiationHessianExtras{G,E2,E2!} <: HessianExtras
    gradient_extras::G
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian(
    f, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)

    hess_var = if backend isa AutoSparse
        sparse_hessian(y_var, x_vec_var)
    else
        hessian(y_var, x_vec_var)
    end
    hess_exe = make_function(hess_var, x_vec_var; in_place=false)
    hess_exe! = make_function(hess_var, x_vec_var; in_place=true)

    gradient_extras = DI.prepare_gradient(f, maybe_dense_ad(backend), x)
    return FastDifferentiationHessianExtras(gradient_extras, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationHessianExtras,
)
    return extras.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationHessianExtras,
)
    extras.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationHessianExtras,
)
    y, grad = DI.value_and_gradient(f, maybe_dense_ad(backend), x, extras.gradient_extras)
    hess = DI.hessian(f, backend, x, extras)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationHessianExtras,
)
    y, _ = DI.value_and_gradient!(
        f, grad, maybe_dense_ad(backend), x, extras.gradient_extras
    )
    DI.hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end

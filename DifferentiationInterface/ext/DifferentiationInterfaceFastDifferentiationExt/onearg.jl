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
    extras::FastDifferentiationOneArgPushforwardExtras,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        if extras.y_prototype isa Number
            return only(extras.jvp_exe(v_vec))
        else
            return reshape(extras.jvp_exe(v_vec), size(extras.y_prototype))
        end
    end
    return ty
end

function DI.pushforward!(
    f,
    ty::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
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
    extras::FastDifferentiationOneArgPushforwardExtras,
    backend::AutoFastDifferentiation,
    x,
    tx::Tangents,
)
    return f(x), DI.pushforward(f, extras, backend, x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    extras::FastDifferentiationOneArgPushforwardExtras,
    backend::AutoFastDifferentiation,
    x,
    tx::Tangents,
)
    return f(x), DI.pushforward!(f, ty, extras, backend, x, tx)
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
    extras::FastDifferentiationOneArgPullbackExtras,
    ::AutoFastDifferentiation,
    x,
    ty::Tangents,
)
    tx = map(ty) do dy
        v_vec = vcat(myvec(x), myvec(dy))
        if x isa Number
            return only(extras.vjp_exe(v_vec))
        else
            return reshape(extras.vjp_exe(v_vec), size(x))
        end
    end
    return tx
end

function DI.pullback!(
    f,
    tx::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
    ::AutoFastDifferentiation,
    x,
    ty::Tangents,
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
    extras::FastDifferentiationOneArgPullbackExtras,
    backend::AutoFastDifferentiation,
    x,
    ty::Tangents,
)
    return f(x), DI.pullback(f, extras, backend, x, ty)
end

function DI.value_and_pullback!(
    f,
    tx::Tangents,
    extras::FastDifferentiationOneArgPullbackExtras,
    backend::AutoFastDifferentiation,
    x,
    ty::Tangents,
)
    return f(x), DI.pullback!(f, tx, extras, backend, x, ty)
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
    f, extras::FastDifferentiationOneArgDerivativeExtras, ::AutoFastDifferentiation, x
)
    if extras.y_prototype isa Number
        return only(extras.der_exe(monovec(x)))
    else
        return reshape(extras.der_exe(monovec(x)), size(extras.y_prototype))
    end
end

function DI.derivative!(
    f, der, extras::FastDifferentiationOneArgDerivativeExtras, ::AutoFastDifferentiation, x
)
    extras.der_exe!(vec(der), monovec(x))
    return der
end

function DI.value_and_derivative(
    f,
    extras::FastDifferentiationOneArgDerivativeExtras,
    backend::AutoFastDifferentiation,
    x,
)
    return f(x), DI.derivative(f, extras, backend, x)
end

function DI.value_and_derivative!(
    f,
    der,
    extras::FastDifferentiationOneArgDerivativeExtras,
    backend::AutoFastDifferentiation,
    x,
)
    return f(x), DI.derivative!(f, der, extras, backend, x)
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
    f, extras::FastDifferentiationOneArgGradientExtras, ::AutoFastDifferentiation, x
)
    jac = extras.jac_exe(vec(x))
    grad_vec = @view jac[1, :]
    return reshape(grad_vec, size(x))
end

function DI.gradient!(
    f, grad, extras::FastDifferentiationOneArgGradientExtras, ::AutoFastDifferentiation, x
)
    extras.jac_exe!(reshape(grad, 1, length(grad)), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, extras::FastDifferentiationOneArgGradientExtras, backend::AutoFastDifferentiation, x
)
    return f(x), DI.gradient(f, extras, backend, x)
end

function DI.value_and_gradient!(
    f,
    grad,
    extras::FastDifferentiationOneArgGradientExtras,
    backend::AutoFastDifferentiation,
    x,
)
    return f(x), DI.gradient!(f, grad, extras, backend, x)
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
    extras::FastDifferentiationOneArgJacobianExtras,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return extras.jac_exe(vec(x))
end

function DI.jacobian!(
    f,
    jac,
    extras::FastDifferentiationOneArgJacobianExtras,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    extras.jac_exe!(jac, vec(x))
    return jac
end

function DI.value_and_jacobian(
    f,
    extras::FastDifferentiationOneArgJacobianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return f(x), DI.jacobian(f, extras, backend, x)
end

function DI.value_and_jacobian!(
    f,
    jac,
    extras::FastDifferentiationOneArgJacobianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return f(x), DI.jacobian!(f, jac, extras, backend, x)
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
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
    ::AutoFastDifferentiation,
    x,
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
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
    backend::AutoFastDifferentiation,
    x,
)
    extras.der2_exe!(vec(der2), monovec(x))
    return der2
end

function DI.value_derivative_and_second_derivative(
    f,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
    backend::AutoFastDifferentiation,
    x,
)
    y, der = DI.value_and_derivative(f, extras.derivative_extras, backend, x)
    der2 = DI.second_derivative(f, extras, backend, x)
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
    backend::AutoFastDifferentiation,
    x,
)
    y, _ = DI.value_and_derivative!(f, der, extras.derivative_extras, backend, x)
    DI.second_derivative!(f, der2, extras, backend, x)
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
    f, extras::FastDifferentiationHVPExtras, ::AutoFastDifferentiation, x, tx::Tangents
)
    dgs = map(tx.d) do dx
        v_vec = vcat(vec(x), vec(dx))
        dg_vec = extras.hvp_exe(v_vec)
        return reshape(dg_vec, size(x))
    end
    return Tangents(dgs)
end

function DI.hvp!(
    f,
    tg::Tangents,
    extras::FastDifferentiationHVPExtras,
    ::AutoFastDifferentiation,
    x,
    tx::Tangents,
)
    for b in eachindex(tx.d, tg.d)
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
    extras::FastDifferentiationHessianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return extras.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    extras::FastDifferentiationHessianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    extras.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    extras::FastDifferentiationHessianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
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
    extras::FastDifferentiationHessianExtras,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    y, _ = DI.value_and_gradient!(
        f, grad, extras.gradient_extras, maybe_dense_ad(backend), x
    )
    DI.hessian!(f, hess, extras, backend, x)
    return y, grad, hess
end

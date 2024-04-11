## Pushforward

struct FastDifferentiationOneArgPushforwardExtras{Y,E1,E2} <: PushforwardExtras
    y_prototype::Y
    jvp_exe::E1
    jvp_exe!::E2
end

function DI.prepare_pushforward(f, ::AnyAutoFastDifferentiation, x)
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
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    v_vec = vcat(myvec(x), myvec(dx))
    if extras.y_prototype isa Number
        return only(extras.jvp_exe(v_vec))
    else
        return reshape(extras.jvp_exe(v_vec), size(extras.y_prototype))
    end
end

function DI.pushforward!(
    f,
    dy,
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.jvp_exe!(vec(dy), v_vec)
    return dy
end

function DI.value_and_pushforward(
    f,
    backend::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end
function DI.value_and_pushforward!(
    f,
    dy,
    backend::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationOneArgPushforwardExtras,
)
    return f(x), DI.pushforward!(f, dy, backend, x, dx, extras)
end

## Pullback

# TODO: this only fails for scalar -> matrix, not sure why

## Derivative

struct FastDifferentiationOneArgDerivativeExtras{Y,E1,E2} <: DerivativeExtras
    y_prototype::Y
    der_exe::E1
    der_exe!::E2
end

function DI.prepare_derivative(f, ::AnyAutoFastDifferentiation, x)
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
    f, ::AnyAutoFastDifferentiation, x, extras::FastDifferentiationOneArgDerivativeExtras
)
    if extras.y_prototype isa Number
        return only(extras.der_exe(monovec(x)))
    else
        return reshape(extras.der_exe(monovec(x)), size(extras.y_prototype))
    end
end

function DI.derivative!(
    f,
    der,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgDerivativeExtras,
)
    extras.der_exe!(vec(der), monovec(x))
    return der
end

function DI.value_and_derivative(
    f,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgDerivativeExtras,
)
    return f(x), DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative!(
    f,
    der,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgDerivativeExtras,
)
    return f(x), DI.derivative!(f, der, backend, x, extras)
end

## Jacobian

struct FastDifferentiationOneArgJacobianExtras{Y,E1,E2} <: JacobianExtras
    y_prototype::Y
    jac_exe::E1
    jac_exe!::E2
end

function DI.prepare_jacobian(f, backend::AnyAutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    if issparse(backend)
        jac_var = sparse_jacobian(y_vec_var, x_vec_var)
    else
        jac_var = jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgJacobianExtras(y_prototype, jac_exe, jac_exe!)
end

function DI.jacobian(
    f, ::AnyAutoFastDifferentiation, x, extras::FastDifferentiationOneArgJacobianExtras
)
    return extras.jac_exe(vec(x))
end

function DI.jacobian!(
    f, jac, ::AnyAutoFastDifferentiation, x, extras::FastDifferentiationOneArgJacobianExtras
)
    extras.jac_exe!(jac, vec(x))
    return jac
end

function DI.value_and_jacobian(
    f,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationOneArgJacobianExtras,
)
    return f(x), DI.jacobian!(f, jac, backend, x, extras)
end

## Second derivative

struct FastDifferentiationAllocatingSecondDerivativeExtras{Y,E1,E2} <:
       SecondDerivativeExtras
    y_prototype::Y
    der2_exe::E1
    der2_exe!::E2
end

function DI.prepare_second_derivative(f, ::AnyAutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    der2_vec_var = derivative(y_vec_var, x_var, x_var)
    der2_exe = make_function(der2_vec_var, x_vec_var; in_place=false)
    der2_exe! = make_function(der2_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationAllocatingSecondDerivativeExtras(
        y_prototype, der2_exe, der2_exe!
    )
end

function DI.second_derivative(
    f,
    ::AnyAutoFastDifferentiation,
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
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    extras.der2_exe!(vec(der2), monovec(x))
    return der2
end

## HVP

struct FastDifferentiationHVPExtras{E1,E2} <: HVPExtras
    hvp_exe::E1
    hvp_exe!::E2
end

function DI.prepare_hvp(f, ::AnyAutoFastDifferentiation, x, v)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    hv_vec_var, v_vec_var = hessian_times_v(y_var, x_vec_var)
    hvp_exe = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    hvp_exe! = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationHVPExtras(hvp_exe, hvp_exe!)
end

function DI.hvp(f, ::AnyAutoFastDifferentiation, x, v, extras::FastDifferentiationHVPExtras)
    v_vec = vcat(vec(x), vec(v))
    hv_vec = extras.hvp_exe(v_vec)
    return reshape(hv_vec, size(x))
end

function DI.hvp!(
    f, p, ::AnyAutoFastDifferentiation, x, v, extras::FastDifferentiationHVPExtras
)
    v_vec = vcat(vec(x), vec(v))
    extras.hvp_exe!(p, v_vec)
    return p
end

## Hessian

struct FastDifferentiationHessianExtras{E1,E2} <: HessianExtras
    hess_exe::E1
    hess_exe!::E2
end

function DI.prepare_hessian(f, backend::AnyAutoFastDifferentiation, x)
    x_vec_var = make_variables(:x, size(x)...)
    y_vec_var = f(x_vec_var)
    if issparse(backend)
        hess_var = sparse_hessian(y_vec_var, vec(x_vec_var))
    else
        hess_var = hessian(y_vec_var, vec(x_vec_var))
    end
    hess_exe = make_function(hess_var, vec(x_vec_var); in_place=false)
    hess_exe! = make_function(hess_var, vec(x_vec_var); in_place=true)
    return FastDifferentiationHessianExtras(hess_exe, hess_exe!)
end

function DI.hessian(
    f, backend::AnyAutoFastDifferentiation, x, extras::FastDifferentiationHessianExtras
)
    return extras.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationHessianExtras,
)
    extras.hess_exe!(hess, vec(x))
    return hess
end

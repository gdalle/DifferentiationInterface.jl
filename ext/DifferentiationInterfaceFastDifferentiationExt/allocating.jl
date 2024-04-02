## Pushforward

struct FastDifferentiationAllocatingPushforwardExtras{E} <: PushforwardExtras
    jvp_exe::E
end

function DI.prepare_pushforward(f, ::AnyAutoFastDifferentiation, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? [x_var] : vec(x_var)
    y_vec_var = y_var isa Number ? [y_var] : vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return FastDifferentiationAllocatingPushforwardExtras(jvp_exe)
end

function DI.value_and_pushforward(
    f,
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationAllocatingPushforwardExtras,
)
    y = f(x)
    v_vec = vcat(myvec(x), myvec(dx))
    jv_vec = extras.jvp_exe(v_vec)
    if y isa Number
        return y, only(jv_vec)
    else
        return y, reshape(jv_vec, size(y))
    end
end

## Pullback

# TODO: this only fails for scalar -> matrix, not sure why

struct FastDifferentiationAllocatingPullbackExtras{E} <: PullbackExtras
    vjp_exe::E
end

function DI.prepare_pullback(f, ::AnyAutoFastDifferentiation, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? [x_var] : vec(x_var)
    y_vec_var = y_var isa Number ? [y_var] : vec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(vj_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return FastDifferentiationAllocatingPullbackExtras(vjp_exe)
end

function DI.value_and_pullback(
    f,
    ::AnyAutoFastDifferentiation,
    x,
    dy,
    extras::FastDifferentiationAllocatingPullbackExtras,
)
    y = f(x)
    v_vec = vcat(myvec(x), myvec(dy))
    vj_vec = extras.vjp_exe(v_vec)
    if x isa Number
        return y, only(vj_vec)
    else
        return y, reshape(vj_vec, size(x))
    end
end

## Derivative

struct FastDifferentiationAllocatingDerivativeExtras{E} <: DerivativeExtras
    der_exe::E
end

function DI.prepare_derivative(f, ::AnyAutoFastDifferentiation, x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = [x_var]
    y_vec_var = y_var isa Number ? [y_var] : vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    return FastDifferentiationAllocatingDerivativeExtras(der_exe)
end

function DI.value_and_derivative(
    f,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingDerivativeExtras,
)
    y = f(x)
    der_vec = extras.der_exe([x])
    if y isa Number
        return y, only(der_vec)
    else
        return y, reshape(der_vec, size(y))
    end
end

function DI.value_and_derivative!!(
    f,
    der,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingDerivativeExtras,
)
    return DI.value_and_derivative(f, backend, x, extras)
end

## Jacobian

struct FastDifferentiationAllocatingJacobianExtras{E} <: JacobianExtras
    jac_exe::E
end

function DI.prepare_jacobian(f, backend::AnyAutoFastDifferentiation, x)
    x_vec_var = make_variables(:x, size(x)...)
    y_vec_var = f(x_vec_var)
    if issparse(backend)
        jac_var = sparse_jacobian(vec(y_vec_var), vec(x_vec_var))
    else
        jac_var = jacobian(vec(y_vec_var), vec(x_vec_var))
    end
    jac_exe = make_function(jac_var, vec(x_vec_var); in_place=false)
    return FastDifferentiationAllocatingJacobianExtras(jac_exe)
end

function DI.jacobian(
    f,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingJacobianExtras,
)
    return extras.jac_exe(vec(x))
end

function DI.value_and_jacobian(
    f,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!!(
    f,
    jac,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingJacobianExtras,
)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(
    f,
    jac,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingJacobianExtras,
)
    return DI.value_and_jacobian(f, backend, x, extras)
end

## Second derivative

struct FastDifferentiationAllocatingSecondDerivativeExtras{E} <: DerivativeExtras
    der2_exe::E
end

function DI.prepare_second_derivative(f, ::AnyAutoFastDifferentiation, x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = [x_var]
    y_vec_var = y_var isa Number ? [y_var] : vec(y_var)
    der2_vec_var = derivative(y_vec_var, x_var, x_var)
    der2_exe = make_function(der2_vec_var, x_vec_var; in_place=false)
    return FastDifferentiationAllocatingSecondDerivativeExtras(der2_exe)
end

function DI.second_derivative(
    f,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    y = f(x)
    der2_vec = extras.der2_exe([x])
    if y isa Number
        return only(der2_vec)
    else
        return reshape(der2_vec, size(y))
    end
end

function DI.second_derivative!!(
    f,
    der2,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationAllocatingSecondDerivativeExtras,
)
    return DI.second_derivative(f, backend, x, extras)
end

## HVP

struct FastDifferentiationHVPExtras{E} <: HVPExtras
    hvp_exe::E
end

function DI.prepare_hvp(f, ::AnyAutoFastDifferentiation, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? [x_var] : vec(x_var)
    hv_vec_var, v_vec_var = hessian_times_v(y_var, x_vec_var)
    hvp_exe = make_function(hv_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return FastDifferentiationHVPExtras(hvp_exe)
end

function DI.hvp(f, ::AnyAutoFastDifferentiation, x, v, extras::FastDifferentiationHVPExtras)
    v_vec = vcat(myvec(x), myvec(v))
    hv_vec = extras.hvp_exe(v_vec)
    if x isa Number
        return only(hv_vec)
    else
        return reshape(hv_vec, size(x))
    end
end

function DI.hvp!!(
    f, p, backend::AnyAutoFastDifferentiation, x, v, extras::FastDifferentiationHVPExtras
)
    return DI.hvp(f, backend, x, v, extras)
end

## Hessian

struct FastDifferentiationHessianExtras{E} <: HessianExtras
    hess_exe::E
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
    return FastDifferentiationHessianExtras(hess_exe)
end

function DI.hessian(
    f, backend::AnyAutoFastDifferentiation, x, extras::FastDifferentiationHessianExtras
)
    return extras.hess_exe(vec(x))
end

function DI.hessian!!(
    f,
    hess,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationHessianExtras,
)
    return DI.hessian(f, backend, x, extras)
end

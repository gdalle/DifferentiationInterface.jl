## Pushforward

struct FastDifferentiationMutatingPushforwardExtras{E} <: PushforwardExtras
    jvp_exe::E
end

function DI.prepare_pushforward(f!, ::AnyAutoFastDifferentiation, y, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? [x_var] : vec(x_var)
    y_vec_var = vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return FastDifferentiationMutatingPushforwardExtras(jvp_exe)
end

function DI.value_and_pushforward!!(
    f!,
    y,
    _dy,
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationMutatingPushforwardExtras,
)
    f!(y, x)
    v_vec = vcat(myvec(x), myvec(dx))
    jv_vec = extras.jvp_exe(v_vec)
    if y isa Number
        return y, only(jv_vec)
    else
        return y, reshape(jv_vec, size(y))
    end
end

## Derivative

struct FastDifferentiationMutatingDerivativeExtras{E} <: DerivativeExtras
    der_exe::E
end

function DI.prepare_derivative(f!, ::AnyAutoFastDifferentiation, y, x)
    x_var = only(make_variables(:x))
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = [x_var]
    y_vec_var = vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    return FastDifferentiationMutatingDerivativeExtras(der_exe)
end

function DI.value_and_derivative!!(
    f!,
    y,
    _der,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingDerivativeExtras,
)
    f!(y, x)
    der_vec = extras.der_exe([x])
    return y, reshape(der_vec, size(y))
end

## Jacobian

struct FastDifferentiationMutatingJacobianExtras{E} <: JacobianExtras
    jac_exe::E
end

function DI.prepare_jacobian(f!, backend::AnyAutoFastDifferentiation, y, x)
    x_var = make_variables(:x, size(x)...)
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    if issparse(backend)
        jac_var = sparse_jacobian(y_vec_var, x_vec_var)
    else
        jac_var = jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    return FastDifferentiationMutatingJacobianExtras(jac_exe)
end

function DI.value_and_jacobian!!(
    f!,
    y,
    _jac,
    backend::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingJacobianExtras,
)
    f!(y, x)
    return y, extras.jac_exe(vec(x))
end

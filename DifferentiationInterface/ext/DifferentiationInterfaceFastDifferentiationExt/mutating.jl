## Pushforward

struct FastDifferentiationMutatingPushforwardExtras{E} <: PushforwardExtras
    jvp_exe!::E
end

function DI.prepare_pushforward(f!, ::AnyAutoFastDifferentiation, y, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe! = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationMutatingPushforwardExtras(jvp_exe!)
end

function DI.value_and_pushforward!!(
    f!,
    y,
    dy,
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationMutatingPushforwardExtras,
)
    f!(y, x)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.jvp_exe!(vec(dy), v_vec)
    return y, dy
end

function DI.pushforward!!(
    f!,
    y,
    dy,
    ::AnyAutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationMutatingPushforwardExtras,
)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.jvp_exe!(vec(dy), v_vec)
    return dy
end

## Derivative

struct FastDifferentiationMutatingDerivativeExtras{E} <: DerivativeExtras
    der_exe!::E
end

function DI.prepare_derivative(f!, ::AnyAutoFastDifferentiation, y, x)
    x_var = only(make_variables(:x))
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe! = make_function(der_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationMutatingDerivativeExtras(der_exe!)
end

function DI.value_and_derivative!!(
    f!,
    y,
    der,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingDerivativeExtras,
)
    f!(y, x)
    extras.der_exe!(der, monovec(x))
    return y, der
end

function DI.derivative!!(
    f!,
    y,
    der,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingDerivativeExtras,
)
    extras.der_exe!(der, monovec(x))
    return der
end

## Jacobian

struct FastDifferentiationMutatingJacobianExtras{E} <: JacobianExtras
    jac_exe!::E
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
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationMutatingJacobianExtras(jac_exe!)
end

function DI.value_and_jacobian!!(
    f!,
    y,
    jac,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingJacobianExtras,
)
    f!(y, x)
    extras.jac_exe!(jac, vec(x))
    return y, jac
end

function DI.jacobian!!(
    f!,
    y,
    jac,
    ::AnyAutoFastDifferentiation,
    x,
    extras::FastDifferentiationMutatingJacobianExtras,
)
    extras.jac_exe!(jac, vec(x))
    return jac
end

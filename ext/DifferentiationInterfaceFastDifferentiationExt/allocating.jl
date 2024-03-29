## Pushforward

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
    return jvp_exe
end

function DI.value_and_pushforward(
    f, ::AnyAutoFastDifferentiation, x, dx, jvp_exe::RuntimeGeneratedFunction
)
    y = f(x)
    v_vec = vcat(myvec(x), myvec(dx))
    jv_vec = jvp_exe(v_vec)
    if y isa Number
        return y, only(jv_vec)
    else
        return y, reshape(jv_vec, size(y))
    end
end

## Pullback

#=

# TODO: this only fails for scalar -> matrix, not sure why

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
    return vjp_exe
end

function DI.value_and_pullback(
    f, ::AnyAutoFastDifferentiation, x, dy, vjp_exe::RuntimeGeneratedFunction
)
    y = f(x)
    v_vec = vcat(myvec(x), myvec(dy))
    vj_vec = vjp_exe(v_vec)
    if x isa Number
        return y, only(vj_vec)
    else
        return y, reshape(vj_vec, size(x))
    end
end

=#

## Jacobian

function DI.prepare_jacobian(f, ::AnyAutoFastDifferentiation, x)
    x_vec_var = make_variables(:x, size(x)...)
    y_vec_var = f(x_vec_var)
    jac_var = jacobian(vec(y_vec_var), vec(x_vec_var))
    jac_exe = make_function(jac_var, vec(x_vec_var); in_place=false)
    return jac_exe
end

function DI.jacobian(
    f, backend::AnyAutoFastDifferentiation, x, jac_exe::RuntimeGeneratedFunction
)
    return jac_exe(vec(x))
end

function DI.value_and_jacobian(f, backend, x, extras)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!!(f, backend::AnyAutoFastDifferentiation, x, extras)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(f, backend::AnyAutoFastDifferentiation, x, extras)
    return DI.value_and_jacobian(f, backend, x, extras)
end

## Hessian

function DI.prepare_hessian(f, ::AnyAutoFastDifferentiation, x)
    x_vec_var = make_variables(:x, size(x)...)
    y_vec_var = f(x_vec_var)
    hess_var = hessian(y_vec_var, vec(x_vec_var))
    hess_exe = make_function(hess_var, vec(x_vec_var); in_place=false)
    return hess_exe
end

function DI.hessian(
    f, backend::AnyAutoFastDifferentiation, x, hess_exe::RuntimeGeneratedFunction
)
    return hess_exe(vec(x))
end

function DI.hessian(f, backend::AnyAutoFastDifferentiation, x, extras::Nothing)
    hess_exe = prepare_hessian(f, backend, x)
    return DI.hessian(f, backend, x, hess_exe)
end

function DI.hessian!!(f, backend::AnyAutoFastDifferentiation, x, extras)
    return DI.hessian(f, backend, x, extras)
end

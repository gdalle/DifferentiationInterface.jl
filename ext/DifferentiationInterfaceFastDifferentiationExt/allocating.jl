## Pushforward

function prepare_pushforward(f::F, ::AutoFastDifferentiation, x) where {F}
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? [x_var] : x_var
    y_vec_var = y_var isa Number ? [y_var] : y_var
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return jvp_exe
end

function DI.value_and_pushforward(f::F, backend::AutoFastDifferentiation, x, dx) where {F}
    y = f(x)
    jvp_exe = prepare_pushforward(f, backend, x)
    v_vec = vcat(myvec(x), myvec(dx))
    jv_vec = jvp_exe(v_vec)
    if y isa Number
        return y, only(jv_vec)
    else
        return y, jv_vec
    end
end

function DI.value_and_pushforward!(
    f::F, dy, backend::AutoFastDifferentiation, x, dx
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx)
    return y, myupdate!(dy, new_dy)
end

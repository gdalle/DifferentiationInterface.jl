struct MyClosure{args,F,X,Y}
    f::F
    x_buffer::Vector{X}
    y_buffer::Vector{Y}
end

function (mc::MyClosure{1})(x)
    mc.x_buffer[1] = x
    mc.y_buffer[1] = mc.f(x)
    return copy(mc.y_buffer[1])
end

function (mc::MyClosure{2})(y, x)
    mc.x_buffer[1] = x
    mc.f(mc.y_buffer[1], mc.x_buffer[1])
    copyto!(y, mc.y_buffer[1])
    return nothing
end

"""
    make_closure(scen::Scenario)

Return a new [`Scenario`](@ref) with a modified function `f` or `f!` that closes over differentiable data.
"""
function make_closure(scen::Scenario)
    @compat (; f, x, y) = scen
    x_buffer = [zero(x)]
    y_buffer = [zero(y)]
    closure_f = MyClosure{nb_args(scen),typeof(f),typeof(x),typeof(y)}(
        f, x_buffer, y_buffer
    )
    return change_function(scen, closure_f)
end

function change_function(scen::Scenario{op,args,pl}, new_f) where {op,args,pl}
    return Scenario{op,args,pl}(
        new_f; x=scen.x, y=scen.y, seed=scen.seed, res1=scen.res1, res2=scen.res2
    )
end

maybe_zero(x::Number) = zero(x)
maybe_zero(x::AbstractArray) = zero(x)
maybe_zero(x::Tangents) = Tangents(map(maybe_zero, x.d)...)
maybe_zero(::Nothing) = nothing

function Base.zero(scen::Scenario{op,args,pl}) where {op,args,pl}
    return Scenario{op,args,pl}(
        scen.f;
        x=scen.x,
        y=scen.y,
        seed=scen.seed,
        res1=maybe_zero(scen.res1),
        res2=maybe_zero(scen.res2),
    )
end

function batchify(scen::Scenario{op,args,pl}) where {op,args,pl}
    @compat (; f, x, y, seed, res1, res2) = scen
    if op == :pushforward || op == :pullback
        new_seed = Tangents(only(seed), -only(seed))
        new_res1 = Tangents(only(res1), -only(res1))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1=new_res1, res2)
    elseif op == :hvp
        new_seed = Tangents(only(seed), -only(seed))
        new_res2 = Tangents(only(res2), -only(res2))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1, res2=new_res2)
    end
end

function add_batched(scens::AbstractVector{<:Scenario})
    batchifiable_scens = filter(s -> operator(s) in (:pushforward, :pullback, :hvp), scens)
    return vcat(scens, batchify.(batchifiable_scens))
end

function remove_batched(scens::AbstractVector{<:Scenario})
    return filter(s -> !isa(s.seed, Tangents), scens)
end

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

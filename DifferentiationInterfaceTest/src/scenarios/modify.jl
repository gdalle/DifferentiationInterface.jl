function change_function(scen::Scenario{op,args,pl}, new_f) where {op,args,pl}
    return Scenario{op,args,pl}(
        new_f;
        x=scen.x,
        y=scen.y,
        seed=scen.seed,
        res1=scen.res1,
        res2=scen.res2,
        contexts=scen.contexts,
    )
end

maybe_zero(x::Number) = zero(x)
maybe_zero(x::AbstractArray) = zero(x)
maybe_zero(x::Tangents) = Tangents(map(maybe_zero, x.d)...)
maybe_zero(::Nothing) = nothing

maybe_multiply(x::Number, a::Number) = a * x
maybe_multiply(x::AbstractArray, a::Number) = a .* x
maybe_multiply(x::Tangents, a::Number) = map(Base.Fix2(maybe_multiply, a), x)
maybe_multiply(::Nothing, a::Number) = nothing

function Base.zero(scen::Scenario{op,args,pl}) where {op,args,pl}
    return Scenario{op,args,pl}(
        scen.f;
        x=scen.x,
        y=scen.y,
        seed=scen.seed,
        res1=maybe_zero(scen.res1),
        res2=maybe_zero(scen.res2),
        contexts=scen.contexts,
    )
end

function batchify(scen::Scenario{op,args,pl}) where {op,args,pl}
    @compat (; f, x, y, seed, res1, res2, contexts) = scen
    if op == :pushforward || op == :pullback
        new_seed = Tangents(only(seed), -only(seed))
        new_res1 = Tangents(only(res1), -only(res1))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1=new_res1, res2, contexts)
    elseif op == :hvp
        new_seed = Tangents(only(seed), -only(seed))
        new_res2 = Tangents(only(res2), -only(res2))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1, res2=new_res2, contexts)
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

struct MultiplyByConstant{args,F}
    f::F
end

function (mc::MultiplyByConstant{1})(x, a)
    y = a * mc.f(x)
    return y
end

function (mc::MultiplyByConstant{2})(y, x, a)
    mc.f(y, x)
    y .*= a
    return nothing
end

function insert_context(scen::Scenario{op,args,pl}) where {op,args,pl}
    @compat (; f,) = scen
    multiply_f = MultiplyByConstant{args,typeof(f)}(f)
    a = 3
    return Scenario{op,args,pl}(
        multiply_f;
        x=scen.x,
        y=maybe_multiply(scen.y, a),
        seed=scen.seed,
        res1=maybe_multiply(scen.res1, a),
        res2=maybe_multiply(scen.res2, a),
        contexts=(Constant(a),),
    )
end

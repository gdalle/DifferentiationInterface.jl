struct FixTail{F,A<:Tuple}
    f::F
    tail_args::A
end

function (ft::FixTail)(args::Vararg{Any,N}) where {N}
    return ft.f(args..., ft.tail_args...)
end

abstract type Context end

struct Constant{T} <: Context
    data::T
end

unwrap(c::Constant) = c.data

function Base.convert(::Type{Constant{T}}, x::Constant) where {T}
    return Constant(convert(T, x.data))
end

Base.convert(::Type{Constant{T}}, x) where {T} = Constant(convert(T, x))

struct Rewrap{C,T}
    contexts::T
    function Rewrap(contexts::Vararg{Context,C}) where {C}
        T = typeof(contexts)
        return new{C,T}(contexts)
    end
end

function (r::Rewrap{C,T})(unannotated_contexts::Vararg{Any,C}) where {C,T}
    return T(unannotated_contexts)
end

with_context(f) = f

function with_context(f::F, contexts::Vararg{Context,N}) where {F,N}
    tail_args = map(unwrap, contexts)
    return FixTail(f, tail_args)
end

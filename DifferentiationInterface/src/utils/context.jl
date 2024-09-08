struct FixTail{F,A<:Tuple}
    f::F
    tail_args::A
end

function (ft::FixTail)(args::Vararg{Any,N}) where N
    return ft.f(args..., ft.tail_args...)
end

abstract type Context end

struct Constant{T} <: Context
    data::T
end

unwrap(c::Constant) = c.data

with_context(f) = f

function with_context(f::F, contexts::Vararg{Context,N}) where {F,N}
    tail_args = map(unwrap, contexts)
    return FixTail(f, tail_args)
end

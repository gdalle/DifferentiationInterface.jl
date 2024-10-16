struct FixTail{F,A<:Tuple}
    f::F
    tail_args::A
end

function (ft::FixTail)(args::Vararg{Any,N}) where {N}
    return ft.f(args..., ft.tail_args...)
end

"""
    Context

Abstract supertype for additional context arguments, which can be passed to differentiation operators after the active input `x` but are not differentiated.

# See also

- [`Constant`](@ref)
- [`Cache`](@ref)
"""
abstract type Context end

"""
    Constant

Concrete type of [`Context`](@ref) argument which is kept constant during differentiation.

Note that an operator can be prepared with an arbitrary value of the constant.
However, same-point preparation must occur with the exact value that will be reused later.

# Example

```jldoctest
julia> using DifferentiationInterface

julia> import ForwardDiff

julia> f(x, c) = c * sum(abs2, x);

julia> gradient(f, AutoForwardDiff(), [1.0, 2.0], Constant(10))
2-element Vector{Float64}:
 20.0
 40.0

julia> gradient(f, AutoForwardDiff(), [1.0, 2.0], Constant(100))
2-element Vector{Float64}:
 200.0
 400.0
```
"""
struct Constant{T} <: Context
    data::T
end

unwrap(c::Constant) = c.data

function Base.convert(::Type{Constant{T}}, x::Constant) where {T}
    return Constant(convert(T, x.data))
end

Base.convert(::Type{Constant{T}}, x) where {T} = Constant(convert(T, x))

"""
    Cache

Concrete type of [`Context`](@ref) argument which can be mutated with active values during differentiation.

The initial values present inside the cache do not matter.
"""
struct Cache{T} <: Context
    data::T
end

unwrap(c::Cache) = c.data

function Base.convert(::Type{Cache{T}}, x::Cache) where {T}
    return Cache(convert(T, x.data))
end

Base.convert(::Type{Cache{T}}, x) where {T} = Cache(convert(T, x))

struct Rewrap{C,T}
    function Rewrap(contexts::Vararg{Context,C}) where {C}
        T = typeof(contexts)
        return new{C,T}()
    end
end

(::Rewrap{0})() = ()

function (r::Rewrap{C,T})(unannotated_contexts::Vararg{Any,C}) where {C,T}
    return T(unannotated_contexts)
end

with_contexts(f) = f

function with_contexts(f::F, contexts::Vararg{Context,N}) where {F,N}
    tail_args = map(unwrap, contexts)
    return FixTail(f, tail_args)
end

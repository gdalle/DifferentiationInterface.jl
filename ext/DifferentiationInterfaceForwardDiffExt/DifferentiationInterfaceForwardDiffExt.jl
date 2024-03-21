module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
using DifferentiationInterface:
    mode, supports_mutation, supports_pushforward, supports_pullback
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
    GradientConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: dot, mul!
using Test: @testset, @test

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, x::Number) where {F} = Tag{F,typeof(x)}
tag_type(::F, x) where {F} = Tag{F,eltype(x)}

make_dual(::Type{T}, x::Number, dx::Number) where {T} = Dual{T}(x, dx)
make_dual(::Type{T}, x, dx) where {T} = Dual{T}.(x, dx)

my_value(::Type{T}, ydual::Number) where {T} = value(T, ydual)
my_value(::Type{T}, ydual) where {T} = value.(T, ydual)

my_value!(::Type{T}, y::Number, ydual::Number) where {T} = value(T, dual)
my_value!(::Type{T}, y, ydual) where {T} = y .= value.(T, dual)

my_derivative(::Type{T}, ydual) where {T} = extract_derivative(T, ydual)

function my_derivative!(::Type{T}, dy::Number, ydual::Number) where {T}
    return extract_derivative(T, dy, ydual)
end

function my_derivative!(::Type{T}, dy, ydual) where {T}
    return extract_derivative!(T, dy, ydual)
end

function DI.value_and_pushforward!(f::F, dy, ::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = my_value(T, ydual)
    dy = my_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward(f::F, ::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = my_value(T, ydual)
    new_dy = my_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(f!::F, y, dy, ::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f!, x)
    xdual = make_dual(T, x, dx)
    ydual = make_dual(T, y, dy)
    f!(ydual, xdual)
    y = my_value!(T, y, ydual)
    dy = my_derivative!(T, dy, ydual)
    return y, dy
end

end # module

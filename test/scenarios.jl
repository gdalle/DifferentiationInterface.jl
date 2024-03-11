using ForwardDiff: ForwardDiff
using LinearAlgebra
using Random: AbstractRNG, randn!
using StableRNGs

## Test scenarios

@kwdef struct Scenario{F,X,Y,D1,D2,D3,D4}
    "function"
    f::F
    "argument"
    x::X
    "primal value"
    y::Y
    "pushforward seed"
    dx::X
    "pullback seed"
    dy::Y
    "pullback result"
    dx_true::X
    "pushforward result"
    dy_true::Y
    "derivative result"
    der_true::D1 = nothing
    "multiderivative result"
    multider_true::D2 = nothing
    "gradient result"
    grad_true::D3 = nothing
    "Jacobian result"
    jac_true::D4 = nothing
end

## Constructors

function make_scenario(rng::AbstractRNG, f, x)
    y = f(x)
    return make_scenario(rng, f, x, y)
end

function make_scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:Number,Y<:Number}
    dx = randn(rng, X)
    dy = randn(rng, Y)
    der_true = ForwardDiff.derivative(f, x)
    dx_true = der_true * dy
    dy_true = der_true * dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, der_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:Number,Y<:AbstractArray}
    dx = randn(rng, X)
    dy = similar(y)
    randn!(rng, dy)
    multider_true = ForwardDiff.derivative(f, x)
    dx_true = dot(multider_true, dy)
    dy_true = multider_true .* dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, multider_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:AbstractArray,Y<:Number}
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, Y)
    grad_true = ForwardDiff.gradient(f, x)
    dx_true = grad_true .* dy
    dy_true = dot(grad_true, dx)
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, grad_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:AbstractArray,Y<:AbstractArray}
    dx = similar(x)
    randn!(rng, dx)
    dy = similar(y)
    randn!(rng, dy)
    jac_true = ForwardDiff.jacobian(f, x)
    dx_true = reshape(transpose(jac_true) * vec(dy), size(x))
    dy_true = reshape(jac_true * vec(dx), size(y))
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

## Access

get_input_type(::Scenario{F,X}) where {F,X} = X
get_output_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

## Seed

rng = StableRNG(63)

## Scenarios

f_scalar_scalar(x::Number)::Number = sin(x)

f_scalar_vector(x::Number)::AbstractVector = [sin(x), sin(2x)]
f_scalar_matrix(x::Number)::AbstractMatrix = hcat([sin(x) cos(x)], [sin(2x) cos(2x)])

f_vector_scalar(x::AbstractVector)::Number = sum(sin, x)
f_matrix_scalar(x::AbstractMatrix)::Number = sum(sin, x)

f_vector_vector(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))
f_vector_matrix(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

f_matrix_vector(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))
f_matrix_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

## All

scenarios = [
    make_scenario(rng, f_scalar_scalar, 1.0),
    make_scenario(rng, f_scalar_vector, 1.0),
    make_scenario(rng, f_scalar_matrix, 1.0),
    make_scenario(rng, f_vector_scalar, [1.0, 2.0]),
    make_scenario(rng, f_matrix_scalar, [1.0 2.0; 3.0 4.0]),
    make_scenario(rng, f_vector_vector, [1.0, 2.0]),
    make_scenario(rng, f_vector_matrix, [1.0, 2.0]),
    make_scenario(rng, f_matrix_vector, [1.0 2.0; 3.0 4.0]),
    make_scenario(rng, f_matrix_matrix, [1.0 2.0; 3.0 4.0]),
];

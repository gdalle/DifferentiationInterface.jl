in_type(::Scenario{F,X}) where {F,X} = X
out_type(::Scenario{F,X,Y}) where {F,X,Y} = Y
mutating(s::Scenario) = s.mutating

## Auto fill

function DT.Scenario(rng::AbstractRNG, f, x::Union{Number,AbstractArray})
    y = f(x)
    return make_scenario(rng, f, x, y; mutating=false)
end

function DT.Scenario(
    rng::AbstractRNG, f!, x::Union{Number,AbstractArray}, s::NTuple{N,<:Integer}
) where {N}
    y = randn(eltype(x), s...)
    f!(y, x)
    return make_scenario(rng, f!, x, y; mutating=true)
end

function make_scenario(rng::AbstractRNG, f, x::Number, y::Number; mutating)
    dx = randn(rng, typeof(x))
    dy = randn(rng, typeof(y))
    der_true = ForwardDiff.derivative(f, x)
    dx_true = der_true * dy
    dy_true = der_true * dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, der_true, mutating)
end

function make_scenario(rng::AbstractRNG, f, x::Number, y::AbstractArray; mutating)
    dx = randn(rng, typeof(x))
    dy = similar(y)
    randn!(rng, dy)
    if mutating
        multider_true = ForwardDiff.derivative(f, y, x)
    else
        multider_true = ForwardDiff.derivative(f, x)
    end
    dx_true = dot(multider_true, dy)
    dy_true = multider_true .* dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, multider_true, mutating)
end

function make_scenario(rng::AbstractRNG, f, x::AbstractArray, y::Number; mutating)
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, typeof(y))
    grad_true = ForwardDiff.gradient(f, x)
    dx_true = grad_true .* dy
    dy_true = dot(grad_true, dx)
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, grad_true, mutating)
end

function make_scenario(rng::AbstractRNG, f, x::AbstractArray, y::AbstractArray; mutating)
    dx = similar(x)
    randn!(rng, dx)
    dy = similar(y)
    randn!(rng, dy)
    if mutating
        jac_true = ForwardDiff.jacobian(f, y, x)
    else
        jac_true = ForwardDiff.jacobian(f, x)
    end
    dx_true = reshape(transpose(jac_true) * vec(dy), size(x))
    dy_true = reshape(jac_true * vec(dx), size(y))
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, jac_true, mutating)
end

## Defaults

f_scalar_scalar(x::Number)::Number = sin(x)

f_scalar_vector(x::Number)::AbstractVector = [sin(x), sin(2x)]

function f!_scalar_vector(y::AbstractVector, x::Number)
    y[1] = sin(x)
    y[2] = sin(2x)
    return nothing
end

f_scalar_matrix(x::Number)::AbstractMatrix = hcat([sin(x) cos(x)], [sin(2x) cos(2x)])

function f!_scalar_matrix(y::AbstractMatrix, x::Number)
    y[1, 1] = sin(x)
    y[2, 1] = cos(x)
    y[1, 2] = sin(2x)
    y[2, 2] = cos(2x)
    return nothing
end

f_vector_scalar(x::AbstractVector)::Number = sum(sin, x)
f_matrix_scalar(x::AbstractMatrix)::Number = sum(sin, x)

f_vector_vector(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function f!_vector_vector(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

f_vector_matrix(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

function f!_vector_matrix(y::AbstractMatrix, x::AbstractVector)
    y[:, 1] .= sin.(x)
    y[:, 2] .= cos.(x)
    return nothing
end

f_matrix_vector(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))

function f!_matrix_vector(y::AbstractVector, x::AbstractMatrix)
    y[1:length(x)] .= sin.(vec(x))
    y[(length(x) + 1):(2length(x))] .= cos.(vec(x))
    return nothing
end

f_matrix_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function f!_matrix_matrix(y::AbstractMatrix, x::AbstractMatrix)
    y[:, 1] .= sin.(vec(x))
    y[:, 2] .= cos.(vec(x))
    return nothing
end

function default_scenarios_allocating(rng::AbstractRNG)
    scenarios = [
        Scenario(rng, f_scalar_scalar, 1.0),
        Scenario(rng, f_scalar_vector, 1.0),
        Scenario(rng, f_scalar_matrix, 1.0),
        Scenario(rng, f_vector_scalar, [1.0, 2.0]),
        Scenario(rng, f_matrix_scalar, [1.0 2.0; 3.0 4.0]),
        Scenario(rng, f_vector_vector, [1.0, 2.0]),
        Scenario(rng, f_vector_matrix, [1.0, 2.0]),
        Scenario(rng, f_matrix_vector, [1.0 2.0; 3.0 4.0]),
        Scenario(rng, f_matrix_matrix, [1.0 2.0; 3.0 4.0]),
    ]
    return scenarios
end

function default_scenarios_mutating(rng::AbstractRNG)
    scenarios = [
        Scenario(rng, f!_scalar_vector, 1.0, (2,)),
        Scenario(rng, f!_scalar_matrix, 1.0, (2, 2)),
        Scenario(rng, f!_vector_vector, [1.0, 2.0], (4,)),
        Scenario(rng, f!_vector_matrix, [1.0, 2.0], (2, 2)),
        Scenario(rng, f!_matrix_vector, [1.0 2.0; 3.0 4.0], (8,)),
        Scenario(rng, f!_matrix_matrix, [1.0 2.0; 3.0 4.0], (4, 2)),
    ]
    return scenarios
end

function DT.default_scenarios(rng::AbstractRNG)
    return vcat(default_scenarios_allocating(rng), default_scenarios_mutating(rng))
end

DT.default_scenarios() = default_scenarios(default_rng())

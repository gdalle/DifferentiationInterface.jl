in_type(::Scenario{F,X}) where {F,X} = X
out_type(::Scenario{F,X,Y}) where {F,X,Y} = Y
is_mutating(s::Scenario) = s.mutating

function derivative_scenarios(scenarios::Vector{<:Scenario})
    filter(scenarios) do scen
        in_type(scen) <: Number && out_type(scen) <: Number
    end
end

function multiderivative_scenarios(scenarios::Vector{<:Scenario})
    filter(scenarios) do scen
        in_type(scen) <: Number && out_type(scen) <: AbstractArray
    end
end

function gradient_scenarios(scenarios::Vector{<:Scenario})
    filter(scenarios) do scen
        in_type(scen) <: AbstractArray && out_type(scen) <: Number
    end
end

function jacobian_scenarios(scenarios::Vector{<:Scenario})
    filter(scenarios) do scen
        in_type(scen) <: AbstractArray && out_type(scen) <: AbstractArray
    end
end

second_derivative_scenarios(scenarios) = derivative_scenarios(scenarios)
hessian_scenarios(scenarios) = gradient_scenarios(scenarios)

for prep in [
    :prepare_pushforward,
    :prepare_pullback,
    :prepare_derivative,
    :prepare_multiderivative,
    :prepare_gradient,
    :prepare_jacobian,
    :prepare_second_derivative,
    :prepare_hessian,
]
    @eval function DI.$prep(ba::AbstractADType, scen::Scenario)
        if is_mutating(scen)
            return DI.$prep(ba, scen.f, scen.x, deepcopy(scen.y))
        else
            return DI.$prep(ba, scen.f, scen.x)
        end
    end
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

function default_scenarios_allocating()
    scenarios = [
        Scenario(f_scalar_scalar, 1.0),
        Scenario(f_scalar_vector, 1.0),
        Scenario(f_scalar_matrix, 1.0),
        Scenario(f_vector_scalar, [1.0, 2.0]),
        Scenario(f_matrix_scalar, [1.0 2.0; 3.0 4.0]),
        Scenario(f_vector_vector, [1.0, 2.0]),
        Scenario(f_vector_matrix, [1.0, 2.0]),
        Scenario(f_matrix_vector, [1.0 2.0; 3.0 4.0]),
        Scenario(f_matrix_matrix, [1.0 2.0; 3.0 4.0]),
    ]
    return scenarios
end

function default_scenarios_mutating()
    scenarios = [
        Scenario(f!_scalar_vector, 1.0, (2,)),
        Scenario(f!_scalar_matrix, 1.0, (2, 2)),
        Scenario(f!_vector_vector, [1.0, 2.0], (4,)),
        Scenario(f!_vector_matrix, [1.0, 2.0], (2, 2)),
        Scenario(f!_matrix_vector, [1.0 2.0; 3.0 4.0], (8,)),
        Scenario(f!_matrix_matrix, [1.0 2.0; 3.0 4.0], (4, 2)),
    ]
    return scenarios
end

function DT.default_scenarios()
    return vcat(default_scenarios_allocating(), default_scenarios_mutating())
end

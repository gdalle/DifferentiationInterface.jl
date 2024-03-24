#=
Constraints on the scenarios:
- non-allocating whenever possible
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

const SCALING_VEC = Vector(1:12)
const SCALING_MAT = Matrix((1:3) .* transpose(1:4))

scalar_to_scalar(x::Number)::Number = sin(x)

function scalar_to_vector(x::Number)::AbstractVector
    return sin.(SCALING_VEC .* x) # output size 12
end

function scalar_to_vector!(y::AbstractVector, x::Number)
    n = length(y)
    y[1:(n ÷ 2)] .= sin.(x)
    y[(n ÷ 2 + 1):n] .= sin.(2x)
    return nothing
end

function scalar_to_matrix(x::Number)::AbstractMatrix
    return sin.(SCALING_MAT .* x)  # output size (3, 4)
end

function scalar_to_matrix!(y::AbstractMatrix, x::Number)
    n, m = size(y)
    y[1:(n ÷ 2), 1:(m ÷ 2)] .= sin.(x)
    y[(n ÷ 2 + 1):n, 1:(m ÷ 2)] .= sin.(2x)
    y[1:(n ÷ 2), ((m ÷ 2) + 1):m] .= cos.(x)
    y[(n ÷ 2 + 1):n, ((m ÷ 2) + 1):m] .= cos.(2x)
    return nothing
end

vector_to_scalar(x::AbstractVector)::Number = sum(sin, x)
matrix_to_scalar(x::AbstractMatrix)::Number = sum(sin, x)

vector_to_vector(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function vector_to_vector!(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

vector_to_matrix(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

function vector_to_matrix!(y::AbstractMatrix, x::AbstractVector)
    y[:, 1] .= sin.(x)
    y[:, 2] .= cos.(x)
    return nothing
end

matrix_to_vector(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))

function matrix_to_vector!(y::AbstractVector, x::AbstractMatrix)
    n = length(x)
    y[1:n] .= sin.(getindex.(Ref(x), 1:n))
    y[(n + 1):(2n)] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

matrix_to_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function matrix_to_matrix!(y::AbstractMatrix, x::AbstractMatrix)
    n = length(x)
    y[:, 1] .= sin.(getindex.(Ref(x), 1:n))
    y[:, 2] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

function default_scenarios_allocating()
    scenarios = [
        Scenario(scalar_to_scalar; x=2.0),
        Scenario(scalar_to_vector; x=2.0),
        Scenario(scalar_to_matrix; x=2.0),
        Scenario(vector_to_scalar; x=Vector{Float64}(1:12)),
        Scenario(matrix_to_scalar; x=Matrix{Float64}(reshape(1:12, 3, 4))),
        Scenario(vector_to_vector; x=Vector{Float64}(1:12)),
        Scenario(vector_to_matrix; x=Vector{Float64}(1:12)),
        Scenario(matrix_to_vector; x=Matrix{Float64}(reshape(1:12, 3, 4))),
        Scenario(matrix_to_matrix; x=Matrix{Float64}(reshape(1:12, 3, 4))),
    ]
    return scenarios
end

function default_scenarios_mutating()
    scenarios = [
        Scenario(scalar_to_vector!; x=2.0, y=zeros(Float64, length(SCALING_VEC))),
        Scenario(scalar_to_matrix!; x=2.0, y=zeros(Float64, size(SCALING_MAT))),
        Scenario(vector_to_vector!; x=Vector{Float64}(1:12), y=zeros(Float64, 24)),
        Scenario(vector_to_matrix!; x=Vector{Float64}(1:12), y=zeros(Float64, 12, 2)),
        Scenario(
            matrix_to_vector!; x=Matrix{Float64}(reshape(1:12, 3, 4)), y=zeros(Float64, 24)
        ),
        Scenario(
            matrix_to_matrix!;
            x=Matrix{Float64}(reshape(1:12, 3, 4)),
            y=zeros(Float64, 12, 2),
        ),
    ]
    return scenarios
end

"""
    default_scenarios()

Create a vector of [`Scenario`](@ref)s for testing differentiation. 
"""
function default_scenarios()
    scenarios = vcat(default_scenarios_allocating(), default_scenarios_mutating())
    return scenarios
end

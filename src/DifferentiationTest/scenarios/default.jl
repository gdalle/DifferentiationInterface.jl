#=
Constraints on the scenarios:
- non-allocating whenever possible
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

## Scalar to scalar

scalar_to_scalar(x::Number)::Number = sin(x)

scalar_to_scalar_derivative(x) = cos(x)
scalar_to_scalar_pushforward(x, dx) = scalar_to_scalar_derivative(x) * dx
scalar_to_scalar_pullback(x, dx) = scalar_to_scalar_derivative(x) * dy
scalar_to_scalar_gradient(x) = scalar_to_scalar_derivative(x)
scalar_to_scalar_second_derivative(x) = -sin(x)
scalar_to_scalar_hvp(x) = scalar_to_scalar_second_derivative(x) * v

## Scalar to array

const SCALING_VEC = Vector(1:12)
const SCALING_MAT = Matrix((1:3) .* transpose(1:4))

function scalar_to_vector(x::Number)::AbstractVector
    return sin.(SCALING_VEC .* x) # output size 12
end

function scalar_to_vector!(y::AbstractVector, x::Number)
    y .= sin.(SCALING_VEC .* x)
    return nothing
end

scalar_to_vector_derivative(x) = SCALING_VEC .* cos.(SCALING_VEC .* x)
scalar_to_vector_pushforward(x, dx) = scalar_to_vector_derivative(x) .* dx
scalar_to_vector_pullback(x, dy) = dot(scalar_to_vector_derivative(x), dy)
scalar_to_vector_second_derivative(x) = -(SCALING_VEC .^2) .* sin.(SCALING_VEC .* x)

function scalar_to_matrix(x::Number)::AbstractMatrix
    return sin.(SCALING_MAT .* x)  # output size (3, 4)
end

function scalar_to_matrix!(y::AbstractMatrix, x::Number)
    y .= sin.(SCALING_MAT .* x)
    return nothing
end

scalar_to_matrix_derivative(x) = SCALING_MAT .* cos.(SCALING_MAT .* x)
scalar_to_matrix_pushforward(x, dx) = scalar_to_matrix_derivative(x) .* dx
scalar_to_matrix_pullback(x, dy) = dot(scalar_to_matrix_derivative(x), dy)
scalar_to_matrix_second_derivative(x) = -(SCALING_MAT .^2) .* sin.(SCALING_MAT .* x)

## Array to scalar

vector_to_scalar(x::AbstractVector)::Number = sum(sin, x)
matrix_to_scalar(x::AbstractMatrix)::Number = sum(sin, x)

array_to_scalar_gradient(x) = cos.(x)
array_to_scalar_pushforward(x, dx) = dot(array_to_scalar_gradient(x), dx)
array_to_scalar_pullback(x, dy) = array_to_scalar_gradient(x) .* dy
array_to_scalar_hessian(x) = Diagonal(-sin.(x))
array_to_scalar_hvp(x, v) = array_to_scalar_hessian(x) * v

## Array to array

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
        Scenario(scalar_to_scalar; x=2.0, ref=scalar_to_scalar_ref()),
        Scenario(scalar_to_vector; x=2.0, ref=scalar_to_vector_ref()),
        Scenario(scalar_to_matrix; x=2.0, ref=scalar_to_matrix_ref()),
        Scenario(vector_to_scalar; x=float.(1:12), ref=vector_to_scalar_ref()),
        Scenario(
            matrix_to_scalar; x=float.(reshape(1:12, 3, 4)), ref=matrix_to_scalar_ref()
        ),
        # Scenario(vector_to_vector; x=Vector{Float64}(1:12)),
        # Scenario(vector_to_matrix; x=Vector{Float64}(1:12)),
        # Scenario(matrix_to_vector; x=Matrix{Float64}(reshape(1:12, 3, 4))),
        # Scenario(matrix_to_matrix; x=Matrix{Float64}(reshape(1:12, 3, 4))),
    ]
    return scenarios
end

function default_scenarios_mutating()
    scenarios = [
        Scenario(scalar_to_vector!; x=2.0, y=zeros(12), ref=scalar_to_vector!_ref()),
        Scenario(scalar_to_matrix!; x=2.0, y=zeros(3, 4), ref=scalar_to_matrix!_ref()),
        # Scenario(vector_to_vector!; x=float.(1:12), y=zeros(24)),
        # Scenario(vector_to_matrix!; x=float.(1:12), y=zeros(12, 2)),
        # Scenario(
        #     matrix_to_vector!; x=float.(reshape(1:12, 3, 4)), y=zeros(24)
        # ),
        # Scenario(
        #     matrix_to_matrix!;
        #     x=float.(reshape(1:12, 3, 4)),
        #     y=zeros(12, 2),
        # ),
    ]
    return scenarios
end

"""
    default_scenarios()

Create a vector of [`Scenario`](@ref)s for testing differentiation. 
"""
function default_scenarios()
    scenarios = vcat(
        default_scenarios_allocating(),  #
        # default_scenarios_mutating(),  #
    )
    return scenarios
end

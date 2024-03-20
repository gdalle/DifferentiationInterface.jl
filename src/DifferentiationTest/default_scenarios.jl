const SCALING_VEC = Vector(1:12)
const SCALING_MAT = Matrix((1:3) .* transpose(1:4))

f_scalar_scalar(x::Number)::Number = sin(x)

function f_scalar_vector(x::Number)::AbstractVector
    return sin.(SCALING_VEC .* x) # output size 12
end

function f!_scalar_vector(y::AbstractVector, x::Number)
    for i in eachindex(y)
        y[i] = sin(i * x)
    end
    return nothing
end

function f_scalar_matrix(x::Number)::AbstractMatrix
    return sin.(SCALING_MAT .* x)  # output size (3, 4)
end

function f!_scalar_matrix(y::AbstractMatrix, x::Number)
    for i in axes(y, 1), j in axes(y, 2)
        y[i, j] = sin(i * j * x)
    end
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
    for i in eachindex(IndexLinear(), x)
        y[i] = sin(x[i])
        y[length(x) + i] = cos(x[i])
    end
    return nothing
end

f_matrix_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function f!_matrix_matrix(y::AbstractMatrix, x::AbstractMatrix)
    for i in eachindex(IndexLinear(), x)
        y[i, 1] = sin(x[i])
        y[i, 2] = cos(x[i])
    end
    return nothing
end

function default_scenarios_allocating()
    scenarios = [
        Scenario(f_scalar_scalar, 2.0),
        Scenario(f_scalar_vector, 2.0),
        Scenario(f_scalar_matrix, 2.0),
        Scenario(f_vector_scalar, Vector{Float64}(1:12)),
        Scenario(f_matrix_scalar, Matrix{Float64}(reshape(1:12, 3, 4))),
        Scenario(f_vector_vector, Vector{Float64}(1:12)),
        Scenario(f_vector_matrix, Vector{Float64}(1:12)),
        Scenario(f_matrix_vector, Matrix{Float64}(reshape(1:12, 3, 4))),
        Scenario(f_matrix_matrix, Matrix{Float64}(reshape(1:12, 3, 4))),
    ]
    return scenarios
end

function default_scenarios_mutating()
    scenarios = [
        Scenario(f!_scalar_vector, 2.0, (12,)),
        Scenario(f!_scalar_matrix, 2.0, (3, 4)),
        Scenario(f!_vector_vector, Vector{Float64}(1:12), (24,)),
        Scenario(f!_vector_matrix, Vector{Float64}(1:12), (12, 2)),
        Scenario(f!_matrix_vector, Matrix{Float64}(reshape(1:12, 3, 4)), (24,)),
        Scenario(f!_matrix_matrix, Matrix{Float64}(reshape(1:12, 3, 4)), (12, 2)),
    ]
    return scenarios
end

function default_scenarios()
    return vcat(default_scenarios_allocating(), default_scenarios_mutating())
end

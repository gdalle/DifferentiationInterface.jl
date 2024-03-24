module DifferentiationInterfaceJLArraysExt

using DifferentiationInterface.DifferentiationTest:
    Scenario,
    vector_to_scalar,
    matrix_to_scalar,
    vector_to_vector,
    vector_to_matrix,
    matrix_to_vector,
    matrix_to_matrix
using JLArrays

const SCALING_JLVEC = jl(Vector(1:12))
const SCALING_JLMAT = jl(Matrix((1:3) .* transpose(1:4)))

function scalar_to_jlvector(x::Number)::JLArray{<:Any,1}
    return sin.(SCALING_JLVEC .* x) # output size 12
end

function scalar_to_jlmatrix(x::Number)::JLArray{<:Any,2}
    return sin.(SCALING_JLMAT .* x)  # output size (3, 4)
end

function gpu_scenarios_allocating()
    scenarios = [
        Scenario(scalar_to_jlvector, 2.0),
        Scenario(scalar_to_jlmatrix, 2.0),
        Scenario(vector_to_scalar, jl(Vector{Float64}(1:12))),
        Scenario(matrix_to_scalar, jl(Matrix{Float64}(reshape(1:12, 3, 4)))),
        Scenario(vector_to_vector, jl(Vector{Float64}(1:12))),
        Scenario(vector_to_matrix, jl(Vector{Float64}(1:12))),
        Scenario(matrix_to_vector, jl(Matrix{Float64}(reshape(1:12, 3, 4)))),
        Scenario(matrix_to_matrix, jl(Matrix{Float64}(reshape(1:12, 3, 4)))),
    ]
    return scenarios
end

end

module DifferentiationInterfaceStaticArraysExt

using DifferentiationInterface.DifferentiationTest:
    Scenario,
    vector_to_scalar,
    matrix_to_scalar,
    vector_to_vector,
    vector_to_matrix,
    matrix_to_vector,
    matrix_to_matrix
using StaticArrays

const SCALING_SVEC = SVector{12}(1:12)
const SCALING_SMAT = SMatrix{3,4}((1:3) .* transpose(1:4))

function scalar_to_svector(x::Number)::SVector
    return sin.(SCALING_SVEC .* x) # output size 12
end

function scalar_to_smatrix(x::Number)::SMatrix
    return sin.(SCALING_SMAT .* x)  # output size (3, 4)
end

function static_scenarios_allocating()
    scenarios = [
        Scenario(scalar_to_svector; x=2.0),
        Scenario(scalar_to_smatrix; x=2.0),
        Scenario(vector_to_scalar; x=SVector{12,Float64}(1:12)),
        Scenario(matrix_to_scalar; x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4))),
        Scenario(vector_to_vector; x=SVector{12,Float64}(1:12)),
        Scenario(vector_to_matrix; x=SVector{12,Float64}(1:12)),
        Scenario(matrix_to_vector; x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4))),
        Scenario(matrix_to_matrix; x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4))),
    ]
    return scenarios
end

end

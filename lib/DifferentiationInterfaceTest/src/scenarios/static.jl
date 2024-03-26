const SCALING_SVEC = SVector{12}(1:12)
const SCALING_SMAT = SMatrix{3,4}((1:3) .* transpose(1:4))

function static_scenarios_allocating()
    return [
        Scenario(
            make_scalar_to_array(SCALING_SVEC); x=2.0, ref=scalar_to_array_ref(SCALING_SVEC)
        ),
        Scenario(
            make_scalar_to_array(SCALING_SMAT); x=2.0, ref=scalar_to_array_ref(SCALING_SMAT)
        ),
        Scenario(array_to_scalar; x=SVector{12,Float64}(1:12), ref=array_to_scalar_ref()),
        Scenario(
            array_to_scalar;
            x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4)),
            ref=array_to_scalar_ref(),
        ),
        Scenario(vector_to_vector; x=SVector{12,Float64}(1:12), ref=vector_to_vector_ref()),
        Scenario(vector_to_matrix; x=SVector{12,Float64}(1:12), ref=vector_to_matrix_ref()),
        Scenario(
            matrix_to_vector;
            x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4)),
            ref=matrix_to_vector_ref(),
        ),
        Scenario(
            matrix_to_matrix;
            x=SMatrix{3,4,Float64}(reshape(1:12, 3, 4)),
            ref=matrix_to_matrix_ref(),
        ),
    ]
end

static_scenarios() = static_scenarios_allocating()

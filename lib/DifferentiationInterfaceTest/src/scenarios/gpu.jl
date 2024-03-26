const SCALING_JLVEC = jl(Vector(1:12))
const SCALING_JLMAT = jl(Matrix((1:3) .* transpose(1:4)))

function gpu_scenarios_allocating()
    return [
        Scenario(
            make_scalar_to_array(SCALING_JLVEC);
            x=2.0,
            ref=scalar_to_array_ref(SCALING_JLVEC),
        ),
        Scenario(
            make_scalar_to_array(SCALING_JLMAT);
            x=2.0,
            ref=scalar_to_array_ref(SCALING_JLMAT),
        ),
        Scenario(array_to_scalar; x=jl(float.(1:12)), ref=array_to_scalar_ref()),
        Scenario(
            array_to_scalar; x=jl(float.(reshape(1:12, 3, 4))), ref=array_to_scalar_ref()
        ),
        Scenario(vector_to_vector; x=jl(float.(1:12)), ref=vector_to_vector_ref()),
        Scenario(vector_to_matrix; x=jl(float.(1:12)), ref=vector_to_matrix_ref()),
        Scenario(
            matrix_to_vector; x=jl(float.(reshape(1:12, 3, 4))), ref=matrix_to_vector_ref()
        ),
        Scenario(
            matrix_to_matrix; x=jl(float.(reshape(1:12, 3, 4))), ref=matrix_to_matrix_ref()
        ),
    ]
end

gpu_scenarios() = gpu_scenarios_allocating()

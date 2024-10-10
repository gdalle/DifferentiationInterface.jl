stack_vec_col(t::NTuple) = stack(vec, t; dims=2)
stack_vec_row(t::NTuple) = stack(vec, t; dims=1)

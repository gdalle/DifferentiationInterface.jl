struct SymbolicsSparsityDetector <: ADTypes.AbstractSparsityDetector end

function ADTypes.jacobian_sparsity(f, x, ::SymbolicsSparsityDetector)
    y = similar(f(x))
    f!(y, x) = copyto!(y, f(x))
    return jacobian_sparsity(f!, y, x)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::SymbolicsSparsityDetector)
    f!_vec(y_vec, x_vec) = f!(reshape(y_vec, size(y)), reshape(x_vec, size(x)))
    return jacobian_sparsity(f!_vec, vec(y), vec(x))
end

function ADTypes.hessian_sparsity(f, x, ::SymbolicsSparsityDetector)
    f_vec(x_vec) = f(reshape(x_vec, size(x)))
    return hessian_sparsity(f_vec, vec(x))
end

struct SymbolicsSparsityDetector <: ADTypes.AbstractSparsityDetector end

function ADTypes.jacobian_sparsity(f, x, ::SymbolicsSparsityDetector)
    return Symbolics.jacobian_sparsity(f, x)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::SymbolicsSparsityDetector)
    return Symbolics.jacobian_sparsity(f!, y, x)
end

function ADTypes.hessian_sparsity(f, x, ::SymbolicsSparsityDetector)
    return Symbolics.hessian_sparsity(f, x)
end

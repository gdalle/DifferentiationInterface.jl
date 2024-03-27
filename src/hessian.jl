## Allocating

"""
    hessian(f, backend, x, [extras]) -> hess
"""
function hessian(f, backend::AbstractADType, x, extras=prepare_hessian(f, backend, x))
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian(f, new_backend, x, new_extras)
end

function hessian(f, backend::SecondOrder, x, extras=prepare_hessian(f, backend, x))
    hess = stack(vec(CartesianIndices(x))) do j
        hess_col_j = hvp(f, backend, x, basis(backend, x, j), extras)
        vec(hess_col_j)
    end
    return Symmetric(hess)
end

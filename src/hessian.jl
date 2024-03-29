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
    return hess
end

"""
    hessian!!(f, hess, backend, x, [extras]) -> hess
"""
function hessian!!(
    f, hess, backend::AbstractADType, x, extras=prepare_hessian(f, backend, x)
)
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian!!(f, hess, new_backend, x, new_extras)
end

function hessian!!(f, hess, backend::SecondOrder, x, extras=prepare_hessian(f, backend, x))
    for (k, j) in enumerate(CartesianIndices(x))
        hess_col_j_old = reshape(view(hess, :, k), size(x))
        hess_col_j_new = hvp!!(f, hess_col_j_old, backend, x, basis(backend, x, j), extras)
        # this allocates
        copyto!(hess_col_j_old, hess_col_j_new)
    end
    return hess
end

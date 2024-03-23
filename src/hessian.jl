"""
    hessian(f, backend, x, [extras]) -> hess
"""
function hessian(
    f::F, backend::AbstractADType, x, extras=prepare_hessian(f, backend, x)
) where {F}
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian(f, new_backend, x, new_extras)
end

function hessian(
    f::F, backend::SecondOrder, x, extras=prepare_hessian(f, backend, x)
) where {F}
    # suboptimal for reverse-over-forward
    gradient_closure(z) = gradient(f, inner(backend), z, inner(extras))
    hess = jacobian(gradient_closure, outer(backend), x, outer(extras))
    return hess
end

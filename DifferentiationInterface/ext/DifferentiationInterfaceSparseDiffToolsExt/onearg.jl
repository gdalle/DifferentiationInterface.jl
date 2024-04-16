struct SparseDiffToolsOneArgJacobianExtras{C} <: JacobianExtras
    cache::C
end

struct SparseDiffToolsHessianExtras{C,E} <: HessianExtras
    inner_gradient_closure::C
    outer_jacobian_extras::E
end

## Jacobian

function DI.prepare_jacobian(f, backend::AnyAutoSparse, x::AbstractArray)
    cache = sparse_jacobian_cache(backend, SymbolicsSparsityDetection(), f, x; fx=f(x))
    return SparseDiffToolsOneArgJacobianExtras(cache)
end

function DI.value_and_jacobian!(
    f, jac, backend::AnyAutoSparse, x, extras::SparseDiffToolsOneArgJacobianExtras
)
    sparse_jacobian!(jac, backend, extras.cache, f, x)
    return f(x), jac
end

function DI.jacobian!(
    f, jac, backend::AnyAutoSparse, x, extras::SparseDiffToolsOneArgJacobianExtras
)
    sparse_jacobian!(jac, backend, extras.cache, f, x)
    return jac
end

function DI.value_and_jacobian(
    f, backend::AnyAutoSparse, x, extras::SparseDiffToolsOneArgJacobianExtras
)
    return f(x), sparse_jacobian(backend, extras.cache, f, x)
end

function DI.jacobian(
    f, backend::AnyAutoSparse, x, extras::SparseDiffToolsOneArgJacobianExtras
)
    return sparse_jacobian(backend, extras.cache, f, x)
end

## Hessian

function DI.prepare_hessian(f, backend::SecondOrder{<:AnyAutoSparse}, x)
    inner_gradient_closure(z) = DI.gradient(f, inner(backend), z)
    outer_jacobian_extras = DI.prepare_jacobian(inner_gradient_closure, outer(backend), x)
    return SparseDiffToolsHessianExtras(inner_gradient_closure, outer_jacobian_extras)
end

function DI.hessian(
    f, backend::SecondOrder{<:AnyAutoSparse}, x, extras::SparseDiffToolsHessianExtras
)
    return DI.jacobian(
        extras.inner_gradient_closure, outer(backend), x, extras.outer_jacobian_extras
    )
end

function DI.hessian!(
    f, hess, backend::SecondOrder{<:AnyAutoSparse}, x, extras::SparseDiffToolsHessianExtras
)
    return DI.jacobian!(
        extras.inner_gradient_closure, hess, outer(backend), x, extras.outer_jacobian_extras
    )
end

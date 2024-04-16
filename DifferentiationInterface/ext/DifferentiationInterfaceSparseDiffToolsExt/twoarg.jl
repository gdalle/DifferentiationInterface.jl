struct SparseDiffToolsTwoArgJacobianExtras{C} <: JacobianExtras
    cache::C
end

## Jacobian

function DI.prepare_jacobian(
    f!, y::AbstractArray, backend::AnyTwoArgAutoSparse, x::AbstractArray
)
    cache = sparse_jacobian_cache(backend, SymbolicsSparsityDetection(), f!, similar(y), x)
    return SparseDiffToolsTwoArgJacobianExtras(cache)
end

function DI.value_and_jacobian(
    f!, y, backend::AnyTwoArgAutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
)
    jac = sparse_jacobian(backend, extras.cache, f!, y, x)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, backend::AnyTwoArgAutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
)
    sparse_jacobian!(jac, backend, extras.cache, f!, y, x)
    f!(y, x)
    return y, jac
end

function DI.jacobian(
    f!, y, backend::AnyTwoArgAutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
)
    jac = sparse_jacobian(backend, extras.cache, f!, y, x)
    return jac
end

function DI.jacobian!(
    f!, y, jac, backend::AnyTwoArgAutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
)
    sparse_jacobian!(jac, backend, extras.cache, f!, y, x)
    return jac
end

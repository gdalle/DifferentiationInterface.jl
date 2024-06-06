maybe_inner(backend::SecondOrder) = inner(backend)
maybe_outer(backend::SecondOrder) = outer(backend)
maybe_inner(backend::AbstractADType) = backend
maybe_outer(backend::AbstractADType) = backend

maybe_dense_ad(backend::AutoSparse) = dense_ad(backend)
maybe_dense_ad(backend::AbstractADType) = backend

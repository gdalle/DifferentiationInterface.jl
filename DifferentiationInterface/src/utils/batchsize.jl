"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.

Returns `Val(1)` for backends which have not overloaded it.
"""
pick_batchsize(::AbstractADType, dimension::Integer) = Val(1)

function pick_batchsize(backend::AutoSparse, dimension::Integer)
    return pick_batchsize(dense_ad(backend), dimension)
end

function pick_jacobian_batchsize(
    ::PushforwardFast, backend::AbstractADType; M::Integer, N::Integer
)
    return pick_batchsize(backend, N)
end

function pick_jacobian_batchsize(
    ::PushforwardSlow, backend::AbstractADType; M::Integer, N::Integer
)
    return pick_batchsize(backend, M)
end

function pick_hessian_batchsize(backend::AbstractADType, N::Integer)
    return pick_batchsize(outer(backend), N)
end

threshold_batchsize(backend::AbstractADType, ::Integer) = backend

"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.

Returns `Val(1)` for backends which have not overloaded it.
"""
pick_batchsize(::AbstractADType, dimension::Integer) = Val(1)

function pick_batchsize(::AutoSparse, dimension::Integer)
    throw(ArgumentError("You should query the batch size of the dense backend."))
end

function pick_batchsize(::MixedMode, dimension::Integer)
    throw(
        ArgumentError("You should query the batch size of the forward or reverse backend.")
    )
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

threshold_batchsize(backend::AbstractADType, ::Integer) = backend

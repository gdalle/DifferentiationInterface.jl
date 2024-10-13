has_fixed_batchsize(::AbstractADType) = true
fixed_batchsize(::AbstractADType) = Val(1)

function adaptive_batchsize end

function pick_batchsize(backend::AbstractADType, a)
    if backend isa SecondOrder
        throw(
            ArgumentError(
                "You should select the batch size for the inner or outer backend of $backend",
            ),
        )
    elseif backend isa AutoSparse
        throw(
            ArgumentError(
                "You should select the batch size for the dense backend of $backend"
            ),
        )
    elseif has_fixed_batchsize(backend)
        return fixed_batchsize(backend)
    else
        return adaptive_batchsize(backend, a)
    end
end

threshold_batchsize(backend::AbstractADType, ::Integer) = backend

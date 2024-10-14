"""
    BatchSizeSettings{B,singlebatch,aligned}

Configuration for the batch size deduced from a backend and a sample array of length `N`.

# Type parameters

- `B::Int`: batch size
- `singlebatch::Bool`: whether `B > N`
- `aligned::Bool`: whether `N % B == 0`

# Fields

- `N::Int`: array length
- `A::Int`: number of batches `A = div(N, B, RoundUp)`
- `B_last::Int`: size of the last batch (if `aligned` is `false`)
"""
struct BatchSizeSettings{B,singlebatch,aligned}
    N::Int
    A::Int
    B_last::Int
end

function BatchSizeSettings{B,singlebatch,aligned}(N::Integer) where {B,singlebatch,aligned}
    A = div(N, B, RoundUp)
    B_last = -1
    return BatchSizeSettings{B,singlebatch,aligned}(N, A, B_last)
end

function BatchSizeSettings(::AbstractADType, N::Integer)
    B = 1
    singlebatch = false
    aligned = true
    return BatchSizeSettings{B,singlebatch,aligned}(N)
end

function BatchSizeSettings(backend::AbstractADType, x::AbstractArray)
    N = length(x)
    return BatchSizeSettings(backend, N)
end

function pick_batchsize(backend::AbstractADType, x_or_N::Union{AbstractArray,Integer})
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
    else
        return BatchSizeSettings(backend, x_or_N)
    end
end

threshold_batchsize(backend::AbstractADType, ::Integer) = backend

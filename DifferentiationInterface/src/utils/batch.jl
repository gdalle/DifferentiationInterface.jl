#=
This heuristic is taken from ForwardDiff.jl.
Source file: https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/prelude.jl
=#

const DEFAULT_BATCHSIZE = 8

"""
    pick_batchsize(input_length)

Pick a reasonable batch size for batched derivative evaluation with an input of length `input_length`.
    
The result cannot be larger than `DEFAULT_BATCHSIZE=$DEFAULT_BATCHSIZE`.
"""
function pick_batchsize(input_length::Integer; threshold::Integer=DEFAULT_BATCHSIZE)
    if input_length <= threshold
        return input_length
    else
        nbatches = round(Int, input_length / threshold, RoundUp)
        return round(Int, input_length / nbatches, RoundUp)
    end
end

struct Batch{B,T}
    elements::NTuple{B,T}
end

Base.length(::Batch{B}) where {B} = B
Base.eltype(::Batch{B,T}) where {B,T} = T

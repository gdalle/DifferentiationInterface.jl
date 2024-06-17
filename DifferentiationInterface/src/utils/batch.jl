"""
    pick_batchsize(input_length)

Pick a reasonable batch size for batched derivative evaluation with an input of length `input_length`.
"""
function pick_batchsize(input_length::Integer; threshold::Integer=2)
    return min(threshold, input_length)
end

struct Batch{B,T}
    elements::NTuple{B,T}
end

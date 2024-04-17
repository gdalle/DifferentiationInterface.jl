#=
This heuristic is taken from ForwardDiff.jl.
Source file: https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/prelude.jl
=#

const DEFAULT_CHUNKSIZE = 8

"""
    pick_chunksize(input_length)

Pick a reasonable chunk size for chunked derivative evaluation with an input of length `input_length`.
    
The result cannot be larger than `DEFAULT_CHUNKSIZE=$DEFAULT_CHUNKSIZE`.
"""
function pick_chunksize(input_length::Integer; threshold::Integer=DEFAULT_CHUNKSIZE)
    if input_length <= threshold
        return input_length
    else
        nchunks = round(Int, input_length / threshold, RoundUp)
        return round(Int, input_length / nchunks, RoundUp)
    end
end

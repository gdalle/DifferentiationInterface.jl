"""
    ChainRulesForwardBackend <: AbstractForwardBackend

Enables the use of forward mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
"""
struct ChainRulesForwardBackend{custom,RC} <: AbstractForwardBackend{custom}
    ruleconfig::RC
end

function Base.show(io::IO, backend::ChainRulesForwardBackend{custom}) where {custom}
    return print(
        io,
        "ChainRulesForwardBackend{$(custom ? "custom" : "fallback")}($(backend.ruleconfig))",
    )
end

"""
    ChainRulesReverseBackend <: AbstractReverseBackend

Enables the use of reverse mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
"""
struct ChainRulesReverseBackend{custom,RC} <: AbstractReverseBackend{custom}
    ruleconfig::RC
end

function Base.show(io::IO, backend::ChainRulesReverseBackend{custom}) where {custom}
    return print(
        io,
        "ChainRulesReverseBackend{$(custom ? "custom" : "fallback")}($(backend.ruleconfig))",
    )
end

"""
    FiniteDiffBackend <: AbstractForwardBackend

Enables the use of [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct FiniteDiffBackend{custom,fdtype} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::FiniteDiffBackend{custom,fdtype}) where {custom,fdtype}
    return print(io, "FiniteDiffBackend{$(custom ? "custom" : "fallback"),$fdtype}()")
end

"""
    EnzymeForwardBackend <: AbstractForwardBackend

Enables the use of [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) in forward mode.
"""
struct EnzymeForwardBackend{custom} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::EnzymeForwardBackend{custom}) where {custom}
    return print(io, "EnzymeForwardBackend{$(custom ? "custom" : "fallback")}()")
end

"""
    EnzymeReverseBackend <: AbstractReverseBackend

Enables the use of [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) in reverse mode.

!!! warning
    This backend only works for scalar output.
"""
struct EnzymeReverseBackend{custom} <: AbstractReverseBackend{custom} end

function Base.show(io::IO, ::EnzymeReverseBackend{custom}) where {custom}
    return print(io, "EnzymeReverseBackend{$(custom ? "custom" : "fallback")}()")
end

"""
    ForwardDiffBackend <: AbstractForwardBackend

Enables the use of [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
"""
struct ForwardDiffBackend{custom} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::ForwardDiffBackend{custom}) where {custom}
    return print(io, "ForwardDiffBackend{$(custom ? "custom" : "fallback")}()")
end

"""
    PolyesterForwardDiffBackend <: AbstractForwardBackend

Enables the use of [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl), falling back on [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) if needed.

!!! warning
    This backend only works when the arrays are vectors.
"""
struct PolyesterForwardDiffBackend{custom,C} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::PolyesterForwardDiffBackend{custom,C}) where {custom,C}
    return print(io, "PolyesterForwardDiffBackend{$(custom ? "custom" : "fallback"),$C}()")
end

"""
    ReverseDiffBackend <: AbstractReverseBackend

Performs autodiff with [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).

!!! warning
    This backend only works for array input.
"""
struct ReverseDiffBackend{custom} <: AbstractReverseBackend{custom} end

function Base.show(io::IO, ::ReverseDiffBackend{custom}) where {custom}
    return print(io, "ReverseDiffBackend{$(custom ? "custom" : "fallback")}()")
end

## Pseudo backends

function ZygoteBackend end

## Limitations

handles_input_type(::ReverseDiffBackend, ::Type{<:Number}) = false
handles_output_type(::EnzymeReverseBackend, ::Type{<:AbstractArray}) = false

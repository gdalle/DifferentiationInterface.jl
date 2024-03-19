
pretty(::AutoZeroForward) = "ZeroForward"
pretty(::AutoZeroReverse) = "ZeroReverse"
pretty(::AutoChainRules) = "ChainRules"
pretty(::AutoDiffractor) = "Diffractor"
pretty(::AutoEnzyme) = "Enzyme"
pretty(::AutoFastDifferentiation) = "FastDifferentiation"
pretty(::AutoFiniteDiff) = "FiniteDiff"
pretty(::AutoForwardDiff) = "ForwardDiff"
pretty(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"
pretty(b::AutoReverseDiff) = "ReverseDiff($(b.compile))"
pretty(::AutoTracker) = "Tracker"
pretty(::AutoZygote) = "Zygote"
pretty(b::AbstractADType) = string(b)

"""
    backend_string(backend)

Return a shorter string than the full object printing from ADTypes.jl.
Might be ambiguous.
"""
function backend_string(backend::AbstractADType)
    bs = pretty(backend)
    if isa(mode(backend), ForwardMode)
        return "$bs (forward)"
    elseif isa(mode(backend), ReverseMode)
        return "$bs (reverse)"
    elseif isa(mode(backend), SymbolicMode)
        return "$bs (symbolic)"
    else
        error("Unknown mode")
    end
end

function backend_string(backend::SecondOrder)
    return "$(backend_string(backend.outer)) / $(backend_string(backend.inner))"
end

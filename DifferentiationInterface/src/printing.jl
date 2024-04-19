backend_string_aux(b::AbstractADType) = string(b)

backend_string_aux(::AutoChainRules) = "ChainRules"
backend_string_aux(::AutoDiffractor) = "Diffractor"
backend_string_aux(::AutoEnzyme) = "Enzyme"
backend_string_aux(::AutoFastDifferentiation) = "FastDifferentiation"
backend_string_aux(::AutoFiniteDiff) = "FiniteDiff"
backend_string_aux(::AutoFiniteDifferences) = "FiniteDifferences"
backend_string_aux(::AutoForwardDiff) = "ForwardDiff"
backend_string_aux(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"
backend_string_aux(b::AutoReverseDiff) = "ReverseDiff$(b.compile ? "{compiled}" : "")"
backend_string_aux(::AutoSymbolics) = "Symbolics"
backend_string_aux(::AutoTapir) = "Tapir"
backend_string_aux(::AutoTracker) = "Tracker"
backend_string_aux(::AutoZygote) = "Zygote"

function backend_string(backend::AbstractADType)
    bs = backend_string_aux(backend)
    if mode(backend) isa ForwardMode
        return "$bs (forward)"
    elseif mode(backend) isa ReverseMode
        return "$bs (reverse)"
    elseif mode(backend) isa SymbolicMode
        return "$bs (symbolic)"
    elseif mode(backend) isa ForwardOrReverseMode
        return "$bs (forward/reverse)"
    else
        error("Unknown mode")
    end
end

backend_string(backend::AutoSparse) = "Sparse $(backend_string(dense_ad(backend)))"

function backend_string(backend::SecondOrder)
    return "$(backend_string(outer(backend))) / $(backend_string(inner(backend)))"
end

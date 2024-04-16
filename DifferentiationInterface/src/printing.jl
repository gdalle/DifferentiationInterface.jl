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

backend_string_aux(::AutoSparseFastDifferentiation) = "FastDifferentiation sparse"
backend_string_aux(::AutoSparseFiniteDiff) = "FiniteDiff sparse"
backend_string_aux(::AutoSparseForwardDiff) = "ForwardDiff sparse"
backend_string_aux(::AutoSparsePolyesterForwardDiff) = "PolyesterForwardDiff sparse"
backend_string_aux(::AutoSparseReverseDiff) = "ReverseDiff sparse"
backend_string_aux(::AutoSparseSymbolics) = "Symbolics sparse"
backend_string_aux(::AutoSparseZygote) = "Zygote sparse"

function backend_string(backend::AbstractADType)
    bs = backend_string_aux(backend)
    if mode(backend) == AbstractFiniteDifferencesMode
        return "$bs (finite)"
    elseif mode(backend) == AbstractForwardMode
        return "$bs (forward)"
    elseif mode(backend) == AbstractReverseMode
        return "$bs (reverse)"
    elseif mode(backend) == AbstractSymbolicDifferentiationMode
        return "$bs (symbolic)"
    else
        error("Unknown mode")
    end
end

function backend_string(backend::SecondOrder)
    return "$(backend_string(outer(backend))) / $(backend_string(inner(backend)))"
end

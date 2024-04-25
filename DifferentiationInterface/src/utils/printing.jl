backend_package_name(b::AbstractADType) = string(b)

backend_package_name(::AutoChainRules) = "ChainRules"
backend_package_name(::AutoDiffractor) = "Diffractor"
backend_package_name(::AutoEnzyme) = "Enzyme"
backend_package_name(::AutoFastDifferentiation) = "FastDifferentiation"
backend_package_name(::AutoFiniteDiff) = "FiniteDiff"
backend_package_name(::AutoFiniteDifferences) = "FiniteDifferences"
backend_package_name(::AutoForwardDiff) = "ForwardDiff"
backend_package_name(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"
backend_package_name(::AutoSymbolics) = "Symbolics"
backend_package_name(::AutoTapir) = "Tapir"
backend_package_name(::AutoTracker) = "Tracker"
backend_package_name(::AutoZygote) = "Zygote"
backend_package_name(::AutoReverseDiff) = "ReverseDiff"

backend_string_aux(b::AbstractADType) = backend_package_name(b)
backend_string_aux(b::AutoReverseDiff) = "ReverseDiff$(b.compile ? "{compiled}" : "")"

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

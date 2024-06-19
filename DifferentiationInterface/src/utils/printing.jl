backend_package_name(b::AbstractADType) = strip(string(b), ['(', ')'])
backend_package_name(b::AutoSparse) = backend_package_name(dense_ad(b))

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

backend_package_name(::AF) where {AF<:AutoForwardFromPrimitive} = string(AF)
backend_package_name(::AR) where {AR<:AutoReverseFromPrimitive} = string(AR)

function backend_str(backend::AbstractADType)
    bs = backend_package_name(backend)
    if mode(backend) isa ForwardMode
        return "$bs (forward)"
    elseif mode(backend) isa ReverseMode
        return "$bs (reverse)"
    elseif mode(backend) isa SymbolicMode
        return "$bs (symbolic)"
    elseif mode(backend) isa ForwardOrReverseMode
        return "$bs (forward or reverse)"
    else
        error("Unknown mode")
    end
end

backend_str(backend::AutoSparse) = "Sparse $(backend_str(dense_ad(backend)))"

function backend_str(backend::SecondOrder)
    return "$(backend_str(outer(backend))) / $(backend_str(inner(backend)))"
end

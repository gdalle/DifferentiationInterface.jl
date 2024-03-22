module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoPolyesterForwardDiff, AutoForwardDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Chunk
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!

## Pushforward

function DI.value_and_pushforward!(
    f::F, dy, ::AutoPolyesterForwardDiff{C}, x, dx
) where {F,C}
    return DI.value_and_pushforward!(f, dy, AutoForwardDiff{C,Nothing}(nothing), x, dx)
end

function DI.value_and_pushforward!(
    f!::F, y, dy, ::AutoPolyesterForwardDiff{C}, x, dx
) where {F,C}
    return DI.value_and_pushforward!(f!, y, dy, AutoForwardDiff{C,Nothing}(nothing), x, dx)
end

function DI.value_and_pushforward(f::F, ::AutoPolyesterForwardDiff{C}, x, dx) where {F,C}
    return DI.value_and_pushforward(f, AutoForwardDiff{C,Nothing}(nothing), x, dx)
end

end # module

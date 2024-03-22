module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoPolyesterForwardDiff, AutoForwardDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Chunk
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!

## Pushforward

function DI.value_and_pushforward!(dy, ::AutoPolyesterForwardDiff{C}, f, x, dx) where {C}
    return DI.value_and_pushforward!(dy, AutoForwardDiff{C,Nothing}(nothing), f, x, dx)
end

function DI.value_and_pushforward!(
    y, dy, ::AutoPolyesterForwardDiff{C}, f!, x, dx
) where {C}
    return DI.value_and_pushforward!(y, dy, AutoForwardDiff{C,Nothing}(nothing), f!, x, dx)
end

function DI.value_and_pushforward(::AutoPolyesterForwardDiff{C}, f, x, dx) where {C}
    return DI.value_and_pushforward(AutoForwardDiff{C,Nothing}(nothing), f, x, dx)
end

end # module

## Pushforward

function DI.value_and_pushforward!!(
    f!, y, dy, ::AutoPolyesterForwardDiff{C}, x, dx, extras::Nothing
) where {C}
    return DI.value_and_pushforward!!(
        f!, y, dy, AutoForwardDiff{C,Nothing}(nothing), x, dx, extras
    )
end

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
    extras::Nothing,
) where {C}
    f!(y, x)
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return y, jac
end

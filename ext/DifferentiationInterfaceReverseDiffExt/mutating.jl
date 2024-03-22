## Pullback

function DI.value_and_pullback!(
    f!::F,
    y::AbstractArray,
    dx::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
) where {F}
    jac = jacobian(f!, y, x)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

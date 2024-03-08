const AutoForwardEnzyme = AutoEnzyme{Val{:forward}}
DI.autodiff_mode(::AutoForwardEnzyme) = Val{:forward}()

## Primitives

function DI.value_and_pushforward!(
    _dy::Y, ::AutoForwardEnzyme, f, x::X, dx
) where {X,Y<:Real}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::AutoForwardEnzyme, f, x::X, dx
) where {X,Y<:AbstractArray}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    dy .= new_dy
    return y, dy
end

## Utilities

function DI.value_and_jacobian(::AutoForwardEnzyme, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(Forward, f, x)
    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1332
    return y, reshape(jac, length(y), length(x))
end

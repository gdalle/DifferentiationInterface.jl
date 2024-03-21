## Primitives

function DI.value_and_pushforward!(
    _dy::Real, backend::AutoForwardEnzyme, f, x, dx, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::AbstractArray, backend::AutoForwardEnzyme, f, x, dx, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    dy .= new_dy
    return y, dy
end

function DI.pushforward!(_dy::Real, backend::AutoForwardEnzyme, f, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    new_dy = only(autodiff(backend.mode, f, DuplicatedNoNeed, Duplicated(x, dx_sametype)))
    return new_dy
end

function DI.pushforward!(
    dy::AbstractArray, backend::AutoForwardEnzyme, f, x, dx, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    new_dy = only(autodiff(backend.mode, f, DuplicatedNoNeed, Duplicated(x, dx_sametype)))
    dy .= new_dy
    return dy
end

function DI.value_and_pushforward(backend::AutoForwardEnzyme, f, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    y, dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, dy
end

function DI.pushforward(backend::AutoForwardEnzyme, f, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    dy = only(autodiff(backend.mode, f, DuplicatedNoNeed, Duplicated(x, dx_sametype)))
    return dy
end

## Utilities

function DI.value_and_jacobian(
    backend::AutoForwardEnzyme, f, x::AbstractArray, extras::Nothing
)
    y = f(x)
    jac = jacobian(backend.mode, f, x)
    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1332
    return y, reshape(jac, length(y), length(x))
end

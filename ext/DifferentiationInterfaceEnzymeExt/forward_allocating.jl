## Pushforward

function DI.value_and_pushforward(
    f::F, backend::AutoForwardEnzyme, x, dx, extras::Nothing
) where {F}
    dx_sametype = convert(typeof(x), copy(dx))
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_dy
end

function DI.value_and_pushforward!(
    f::F, dy, backend::AutoForwardEnzyme, x, dx, extras
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, myupdate!(dy, new_dy)
end

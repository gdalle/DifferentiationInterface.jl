## Pushforward

function DI.value_and_pushforward(f, backend::AutoForwardEnzyme, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), copy(dx))
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_dy
end

## Pushforward

function DI.value_and_pushforward(backend::AutoForwardEnzyme, f, x, dx)
    dx_sametype = convert(typeof(x), copy(dx))
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_y
end

function DI.value_and_pushforward!(dy, backend::AutoForwardEnzyme, f, x, dx)
    y, new_dy = DI.value_and_pushforward(backend, f, x, dx)
    return y, myupdate!(dy, new_dy)
end

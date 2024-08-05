struct Bufferize{F,Y}
    f!::F
    y::Y
end

function (f::Bufferize)(x)
    @compat (; f!, y) = f
    buf = Buffer(y)
    f!(buf, x)
    return copy(buf)
end

## Pullback

DI.prepare_pullback(f!, y, ::AutoZygote, x, dy) = NoPullbackExtras()

function DI.value_and_pullback(f!, y, backend::AutoZygote, x, dy, ::NoPullbackExtras)
    f = Bufferize(f!, y)
    y_new, dx = DI.value_and_pullback(f, backend, x, dy)
    copyto!(y, y_new)
    return y, dx
end

## Pullback

function DI.value_and_pullback!!(
    f, _dx, ::AutoReverseEnzyme, x::Number, dy::Number, extras::Nothing
)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!!(f, dx, ::AutoReverseEnzyme, x, dy::Number, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype = myzero!!(dx_sametype)
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx_sametype))
    dx_sametype = mymul!!(dx_sametype, dy)
    return y, myupdate!!(dx, dx_sametype)
end

function DI.value_and_pullback(f, backend::AutoReverseEnzyme, x, dy::Number, extras)
    dx = mysimilar(x)
    return DI.value_and_pullback!!(f, dx, backend, x, dy, extras)
end

## Gradient

function DI.gradient(f, backend::AutoReverseEnzyme, x, extras::Nothing)
    return gradient(Reverse, f, x)
end

function DI.gradient!!(f, grad, backend::AutoReverseEnzyme, x, extras::Nothing)
    return gradient!(Reverse, grad, f, x)
end

function DI.gradient(f, backend::AutoReverseEnzyme, x::Number, extras::Nothing)
    return autodiff(Reverse, f, Active(x))
end

function DI.gradient!!(f, grad, backend::AutoReverseEnzyme, x::Number, extras::Nothing)
    return autodiff(Reverse, f, Active(x))
end

## Pullback

function DI.value_and_pullback!!(
    f, _dx, ::AutoReverseEnzyme, x::Number, dy::Number, extras::Nothing
)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!!(
    f, dx, ::AutoReverseEnzyme, x::AbstractArray, dy::Number, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype = myzero!!(dx_sametype)
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx_sametype))
    dx_sametype = mymul!!(dx_sametype, dy)
    return y, myupdate!!(dx, dx_sametype)
end

function DI.value_and_pullback!!(
    f, _dx, ::AutoReverseEnzyme, x::Number, dy::AbstractArray, extras::Nothing
)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, Active{typeof(x)}
    )
    tape, y, new_dy = forw(Const(f), Active(x))
    new_dy .= dy
    new_dx = only(only(rev(Const(f), Active(x), tape)))
    return y, new_dx
end

function DI.value_and_pullback!!(
    f, dx, ::AutoReverseEnzyme, x::AbstractArray, dy::AbstractArray, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype = myzero!!(dx_sametype)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, Duplicated{typeof(x)}
    )
    tape, y, new_dy = forw(Const(f), Duplicated(x, dx_sametype))
    new_dy .= dy
    rev(Const(f), Duplicated(x, dx_sametype), tape)
    return y, myupdate!!(dx, dx_sametype)
end

function DI.value_and_pullback(f, backend::AutoReverseEnzyme, x, dy, extras)
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

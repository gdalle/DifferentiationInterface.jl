## Pullback

function DI.value_and_pullback!(
    f!::F, y, _dx, ::AutoReverseEnzyme, x::Number, dy, extras::Nothing
) where {F}
    dy_sametype = convert(typeof(y), copy(dy))
    _, new_dx = only(autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Active(x)))
    return y, new_dx
end

function DI.value_and_pullback!(
    f!::F, y, dx, ::AutoReverseEnzyme, x, dy, extras::Nothing
) where {F}
    dx_sametype = convert(typeof(x), dx)
    dx_sametype = myzero!(dx_sametype)
    dy_sametype = convert(typeof(y), copy(dy))
    autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype))
    return y, myupdate!(dx, dx_sametype)
end

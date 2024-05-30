## Pullback

function DI.prepare_pullback(f!, y, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, dy)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    dy,
    ::NoPullbackExtras,
)
    dy_sametype = convert(typeof(y), copy(dy))
    y_and_dy = Duplicated(y, dy_sametype)
    _, new_dx = if backend isa AutoDeferredEnzyme
        only(autodiff_deferred(reverse_mode(backend), f!, Const, y_and_dy, Active(x)))
    else
        only(autodiff(reverse_mode(backend), Const(f!), Const, y_and_dy, Active(x)))
    end
    return y, new_dx
end

function DI.value_and_pullback(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy,
    ::NoPullbackExtras,
)
    dx_sametype = make_zero(x)
    dy_sametype = convert(typeof(y), copy(dy))
    y_and_dy = Duplicated(y, dy_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(reverse_mode(backend), f!, Const, y_and_dy, x_and_dx)
    else
        autodiff(reverse_mode(backend), Const(f!), Const, y_and_dy, x_and_dx)
    end
    return y, dx_sametype
end

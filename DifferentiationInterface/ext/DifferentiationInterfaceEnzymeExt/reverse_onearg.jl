## Pullback

struct ReverseEnzymePullbackExtras{F}
    df::F
end

function DI.prepare_pullback(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, dy)
    df = make_zero(f)
    return ReverseEnzymePullbackExtras(df)
end

### Out-of-place

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    dy::Number,
    extras::ReverseEnzymePullbackExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    der, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, Duplicated(f, df), Active, Active(x))
    else
        autodiff(ReverseWithPrimal, Duplicated(f, df), Active, Active(x))
    end
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    dy::AbstractArray,
    extras::ReverseEnzymePullbackExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    tf, tx = typeof(f), typeof(x)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Duplicated{tf}, Duplicated, Active{tx}
    )
    tape, y, new_dy = forw(Duplicated(f, df), Active(x))
    copyto!(new_dy, dy)
    new_dx = only(only(rev(Duplicated(f, df), Active(x), tape)))
    return y, new_dx
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy,
    extras::ReverseEnzymePullbackExtras,
)
    dx = similar(x)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)
end

function DI.pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    dy,
    extras::ReverseEnzymePullbackExtras,
)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

### In-place

function DI.value_and_pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy::Number,
    extras::ReverseEnzymePullbackExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(x))
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, Duplicated(f, df), Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, Duplicated(f, df), Active, x_and_dx)
    end
    dx_sametype .*= dy
    return y, copyto!(dx, dx_sametype)
end

function DI.value_and_pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy::AbstractArray,
    extras::ReverseEnzymePullbackExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    tf, tx = typeof(f), typeof(x)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Duplicated{tf}, Duplicated, Duplicated{tx}
    )
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(x))
    tape, y, new_dy = forw(Duplicated(f, df), Duplicated(x, dx_sametype))
    copyto!(new_dy, dy)
    rev(Duplicated(f, df), Duplicated(x, dx_sametype), tape)
    return y, copyto!(dx, dx_sametype)
end

function DI.pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    dy,
    extras::ReverseEnzymePullbackExtras,
)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Gradient

struct ReverseEnzymeGradientExtras{F}
    df::F
end

function DI.prepare_gradient(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x)
    df = make_zero(f)
    return ReverseEnzymeGradientExtras(df)
end

function DI.gradient(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    extras::ReverseEnzymeGradientExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    if backend isa AutoDeferredEnzyme
        grad = make_zero(x)
        autodiff_deferred(
            reverse_mode(backend), Duplicated(f, df), Active, Duplicated(x, grad)
        )
        return grad
    else
        return gradient(reverse_mode(backend), Duplicated(f, df), x)
    end
end

function DI.gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    extras::ReverseEnzymeGradientExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    grad_sametype = convert(typeof(x), grad)
    grad_sametype .= zero(eltype(x))
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(
            reverse_mode(backend), Duplicated(f, df), Active, Duplicated(x, grad_sametype)
        )
    else
        gradient!(reverse_mode(backend), grad_sametype, Duplicated(f, df), x)
    end
    return copyto!(grad, grad_sametype)
end

function DI.value_and_gradient(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ::ReverseEnzymeGradientExtras,
)
    return DI.value_and_pullback(f, backend, x, one(eltype(x)), NoPullbackExtras())
end

function DI.value_and_gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ::ReverseEnzymeGradientExtras,
)
    return DI.value_and_pullback!(f, grad, backend, x, one(eltype(x)), NoPullbackExtras())
end

## Jacobian

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1391

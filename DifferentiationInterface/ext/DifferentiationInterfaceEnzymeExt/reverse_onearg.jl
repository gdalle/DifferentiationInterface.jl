## Pullback

function DI.prepare_pullback(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true}, x, dy)
    return NoPullbackExtras()
end

function DI.prepare_pullback(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},false}, x, dy)
    throw(ArgumentError(CONSTANT_FUNCTION_ERROR))
end

### Out-of-place

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x::Number,
    dy::Number,
    ::NoPullbackExtras,
)
    der, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, f, Active, Active(x))
    else
        autodiff(ReverseWithPrimal, Const(f), Active, Active(x))
    end
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x::Number,
    dy,
    ::NoPullbackExtras,
)
    tf, tx = typeof(f), typeof(x)
    forw, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{tf}, Duplicated, Active{tx})
    tape, y, new_dy = forw(Const(f), Active(x))
    copyto!(new_dy, dy)
    new_dx = only(only(rev(Const(f), Active(x), tape)))
    return y, new_dx
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy::Number,
    ::NoPullbackExtras,
)
    dx_sametype = make_zero(x)
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, Const(f), Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, Const(f), Active, x_and_dx)
    end
    if !isone(dy)
        # TODO: generalize beyond Arrays?
        dx_sametype .*= dy
    end
    return y, dx_sametype
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy,
    extras::NoPullbackExtras,
)
    dx = make_zero(x)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)
end

function DI.pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy,
    extras::NoPullbackExtras,
)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

### In-place

function DI.value_and_pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy::Number,
    ::NoPullbackExtras,
)
    dx_sametype = convert(typeof(x), dx)
    make_zero!(dx_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, Const(f), Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, Const(f), Active, x_and_dx)
    end
    if !isone(dy)
        # TODO: generalize beyond Arrays?
        dx_sametype .*= dy
    end
    return y, copyto!(dx, dx_sametype)
end

function DI.value_and_pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy,
    ::NoPullbackExtras,
)
    tf, tx = typeof(f), typeof(x)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{tf}, Duplicated, Duplicated{tx}
    )
    dx_sametype = convert(typeof(x), dx)
    make_zero!(dx_sametype)
    tape, y, new_dy = forw(Const(f), Duplicated(x, dx_sametype))
    copyto!(new_dy, dy)
    rev(Const(f), Duplicated(x, dx_sametype), tape)
    return y, copyto!(dx, dx_sametype)
end

function DI.pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    dy,
    extras::NoPullbackExtras,
)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Gradient

function DI.prepare_gradient(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true}, x)
    return NoGradientExtras()
end

function DI.gradient(
    f, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true}, x, ::NoGradientExtras
)
    if backend isa AutoDeferredEnzyme
        grad = make_zero(x)
        autodiff_deferred(reverse_mode(backend), f, Active, Duplicated(x, grad))
        return grad
    else
        return gradient(reverse_mode(backend), f, x)
    end
end

function DI.gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    extras::NoGradientExtras,
)
    grad_sametype = convert(typeof(x), grad)
    make_zero!(grad_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(reverse_mode(backend), f, Active, Duplicated(x, grad_sametype))
    else
        gradient!(reverse_mode(backend), grad_sametype, f, x)
    end
    return copyto!(grad, grad_sametype)
end

function DI.value_and_gradient(
    f, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true}, x, ::NoGradientExtras
)
    return DI.value_and_pullback(f, backend, x, true, NoPullbackExtras())
end

function DI.value_and_gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},true},
    x,
    ::NoGradientExtras,
)
    return DI.value_and_pullback!(f, grad, backend, x, true, NoPullbackExtras())
end

## Jacobian

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1391

## Pullback

function DI.prepare_pullback(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, dy)
    return NoPullbackExtras()
end

### Out-of-place

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
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
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    dy::AbstractArray,
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
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy,
    extras::NoPullbackExtras,
)
    dx = similar(x)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)
end

function DI.pullback(
    f, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, dy, extras::NoPullbackExtras
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
    ::NoPullbackExtras,
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(x))
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, Const(f), Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, Const(f), Active, x_and_dx)
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
    ::NoPullbackExtras,
)
    tf, tx = typeof(f), typeof(x)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{tf}, Duplicated, Duplicated{tx}
    )
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(x))
    tape, y, new_dy = forw(Const(f), Duplicated(x, dx_sametype))
    copyto!(new_dy, dy)
    rev(Const(f), Duplicated(x, dx_sametype), tape)
    return y, copyto!(dx, dx_sametype)
end

function DI.pullback!(
    f,
    dx,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    dy,
    extras::NoPullbackExtras,
)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Gradient

function DI.prepare_gradient(f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x)
    return NoGradientExtras()
end

function DI.gradient(
    f, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ::NoGradientExtras
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
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    extras::NoGradientExtras,
)
    grad_sametype = convert(typeof(x), grad)
    grad_sametype .= zero(eltype(x))
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(reverse_mode(backend), f, Active, Duplicated(x, grad_sametype))
    else
        gradient!(reverse_mode(backend), grad_sametype, f, x)
    end
    return copyto!(grad, grad_sametype)
end

function DI.value_and_gradient(
    f, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ::NoGradientExtras
)
    return DI.value_and_pullback(f, backend, x, one(eltype(x)), NoPullbackExtras())
end

function DI.value_and_gradient!(
    f, grad, backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ::NoGradientExtras
)
    return DI.value_and_pullback!(f, grad, backend, x, one(eltype(x)), NoPullbackExtras())
end

## Jacobian

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1391

#=

struct EnzymeReverseOneArgJacobianExtras{B,N} end

function DI.prepare_jacobian(f, ::AutoReverseEnzyme, x)
    B = pick_batchsize(length(x))
    y = f(x)
    N = length(y)
    return EnzymeReverseOneArgJacobianExtras{B,N}()
end

function DI.jacobian(
    f,
    backend::AutoReverseEnzyme,
    x::AbstractArray,
    ::EnzymeReverseOneArgJacobianExtras{C,N},
) where {B,N}
    jac_wrongshape = jacobian(reverse_mode(backend), f, x, Val{N}(), Val{B}())
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    jac_rightshape = reshape(jac_wrongshape, ny, nx)
    return jac_rightshape
end

function DI.value_and_jacobian(
    f,
    backend::AutoReverseEnzyme,
    x::AbstractArray,
    extras::EnzymeReverseOneArgJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!(
    f,
    jac,
    backend::AutoReverseEnzyme,
    x::AbstractArray,
    extras::EnzymeReverseOneArgJacobianExtras,
)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::AutoReverseEnzyme,
    x::AbstractArray,
    extras::EnzymeReverseOneArgJacobianExtras,
)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

=#

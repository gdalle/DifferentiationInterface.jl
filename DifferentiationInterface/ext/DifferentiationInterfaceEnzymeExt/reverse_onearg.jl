## Pullback

DI.prepare_pullback(f, ::AutoReverseEnzyme, x, dy) = NoPullbackExtras()

### Out-of-place

function DI.value_and_pullback(
    f, ::AutoReverseEnzyme, x::Number, dy::Number, ::NoPullbackExtras
)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback(
    f, ::AutoReverseEnzyme, x::Number, dy::AbstractArray, ::NoPullbackExtras
)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, Active{typeof(x)}
    )
    tape, y, new_dy = forw(Const(f), Active(x))
    copyto!(new_dy, dy)
    new_dx = only(only(rev(Const(f), Active(x), tape)))
    return y, new_dx
end

function DI.value_and_pullback(
    f, backend::AutoReverseEnzyme, x::AbstractArray, dy, extras::NoPullbackExtras
)
    dx = similar(x)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)
end

function DI.pullback(f, backend::AutoReverseEnzyme, x, dy, extras::NoPullbackExtras)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

### In-place

function DI.value_and_pullback!(
    f, dx, ::AutoReverseEnzyme, x::AbstractArray, dy::Number, ::NoPullbackExtras
)
    dx_sametype = zero_sametype!(dx, x)
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx_sametype))
    dx_sametype .*= dy
    return y, copyto!(dx, dx_sametype)
end

function DI.value_and_pullback!(
    f, dx, ::AutoReverseEnzyme, x::AbstractArray, dy::AbstractArray, ::NoPullbackExtras
)
    dx_sametype = zero_sametype!(dx, x)
    forw, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, Duplicated{typeof(x)}
    )
    tape, y, new_dy = forw(Const(f), Duplicated(x, dx_sametype))
    copyto!(new_dy, dy)
    rev(Const(f), Duplicated(x, dx_sametype), tape)
    return y, copyto!(dx, dx_sametype)
end

function DI.pullback!(f, dx, backend::AutoReverseEnzyme, x, dy, extras::NoPullbackExtras)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Gradient

DI.prepare_gradient(f, ::AutoReverseEnzyme, x) = NoGradientExtras()

function DI.gradient(f, ::AutoReverseEnzyme, x, ::NoGradientExtras)
    return gradient(Reverse, f, x)
end

function DI.gradient!(f, grad, ::AutoReverseEnzyme, x, ::NoGradientExtras)
    grad_sametype = convert(typeof(x), grad)
    gradient!(Reverse, grad_sametype, f, x)
    return copyto!(grad, grad_sametype)
end

function DI.value_and_gradient(f, backend::AutoReverseEnzyme, x, ::NoGradientExtras)
    return DI.value_and_pullback(f, backend, x, one(eltype(x)), NoPullbackExtras())
end

function DI.value_and_gradient!(f, grad, backend::AutoReverseEnzyme, x, ::NoGradientExtras)
    return DI.value_and_pullback!(f, grad, backend, x, one(eltype(x)), NoPullbackExtras())
end

## Jacobian

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1391

#=

struct EnzymeReverseOneArgJacobianExtras{C,N} end

function DI.prepare_jacobian(f, ::AutoReverseEnzyme, x)
    C = pick_chunksize(length(x))
    y = f(x)
    N = length(y)
    return EnzymeReverseOneArgJacobianExtras{C,N}()
end

function DI.jacobian(
    f,
    backend::AutoReverseEnzyme,
    x::AbstractArray,
    ::EnzymeReverseOneArgJacobianExtras{C,N},
) where {C,N}
    jac_wrongshape = jacobian(backend.mode, f, x, Val{N}(), Val{C}())
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

## Pushforward

DI.prepare_pushforward(f, ::AutoForwardOrNothingEnzyme, x, dx) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f, backend::AutoForwardOrNothingEnzyme, x, dx, ::NoPushforwardExtras
)
    dx_sametype = convert(typeof(x), dx)
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_dy
end

function DI.pushforward(
    f, backend::AutoForwardOrNothingEnzyme, x, dx, ::NoPushforwardExtras
)
    dx_sametype = convert(typeof(x), dx)
    new_dy = only(autodiff(backend.mode, f, DuplicatedNoNeed, Duplicated(x, dx_sametype)))
    return new_dy
end

function DI.value_and_pushforward!(
    f, dy, backend::AutoForwardOrNothingEnzyme, x, dx, extras::NoPushforwardExtras
)
    # dy cannot be passed anyway
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function DI.pushforward!(
    f, dy, backend::AutoForwardOrNothingEnzyme, x, dx, extras::NoPushforwardExtras
)
    # dy cannot be passed anyway
    return copyto!(dy, DI.pushforward(f, backend, x, dx, extras))
end

## Gradient

struct EnzymeForwardGradientExtras{C,O}
    shadow::O
end

function DI.prepare_gradient(f, ::AutoForwardEnzyme, x)
    C = pick_chunksize(length(x))
    shadow = chunkedonehot(x, Val(C))
    return EnzymeForwardGradientExtras{C,typeof(shadow)}(shadow)
end

function DI.gradient(
    f, backend::AutoForwardEnzyme, x, extras::EnzymeForwardGradientExtras{C}
) where {C}
    grad_tup = gradient(backend.mode, f, x, Val{C}(); shadow=extras.shadow)
    return reshape(collect(grad_tup), size(x))
end

function DI.value_and_gradient(
    f, backend::AutoForwardEnzyme, x, extras::EnzymeForwardGradientExtras
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!(
    f, grad, backend::AutoForwardEnzyme, x, extras::EnzymeForwardGradientExtras{C}
) where {C}
    grad_tup = gradient(backend.mode, f, x, Val{C}(); shadow=extras.shadow)
    return copyto!(grad, grad_tup)
end

function DI.value_and_gradient!(
    f, grad, backend::AutoForwardEnzyme, x, extras::EnzymeForwardGradientExtras{C}
) where {C}
    grad_tup = gradient(backend.mode, f, x, Val{C}(); shadow=extras.shadow)
    return f(x), copyto!(grad, grad_tup)
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{C,O}
    shadow::O
end

function DI.prepare_jacobian(f, ::AutoForwardOrNothingEnzyme, x)
    C = pick_chunksize(length(x))
    shadow = chunkedonehot(x, Val(C))
    return EnzymeForwardOneArgJacobianExtras{C,typeof(shadow)}(shadow)
end

function DI.jacobian(
    f, backend::AutoForwardOrNothingEnzyme, x, extras::EnzymeForwardOneArgJacobianExtras{C}
) where {C}
    jac_wrongshape = jacobian(backend.mode, f, x, Val{C}(); shadow=extras.shadow)
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    return reshape(jac_wrongshape, ny, nx)
end

function DI.value_and_jacobian(
    f, backend::AutoForwardOrNothingEnzyme, x, extras::EnzymeForwardOneArgJacobianExtras
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!(
    f,
    jac,
    backend::AutoForwardOrNothingEnzyme,
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::AutoForwardOrNothingEnzyme,
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

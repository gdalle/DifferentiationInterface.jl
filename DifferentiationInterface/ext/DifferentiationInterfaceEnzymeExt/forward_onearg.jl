## Pushforward

function DI.prepare_pushforward(f, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f, backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx, ::NoPushforwardExtras
)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    y, new_dy = if backend isa AutoDeferredEnzyme
        autodiff_deferred(forward_mode(backend), f, Duplicated, x_and_dx)
    else
        autodiff(forward_mode(backend), Const(f), Duplicated, x_and_dx)
    end
    return y, new_dy
end

function DI.pushforward(
    f, backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx, ::NoPushforwardExtras
)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    new_dy = if backend isa AutoDeferredEnzyme
        only(autodiff_deferred(forward_mode(backend), f, DuplicatedNoNeed, x_and_dx))
    else
        only(autodiff(forward_mode(backend), Const(f), DuplicatedNoNeed, x_and_dx))
    end
    return new_dy
end

function DI.value_and_pushforward!(
    f,
    dy,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    extras::NoPushforwardExtras,
)
    # dy cannot be passed anyway
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function DI.pushforward!(
    f,
    dy,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    extras::NoPushforwardExtras,
)
    # dy cannot be passed anyway
    return copyto!(dy, DI.pushforward(f, backend, x, dx, extras))
end

## Gradient

struct EnzymeForwardGradientExtras{B,O} <: GradientExtras
    shadow::O
end

function DI.prepare_gradient(f, backend::AutoEnzyme{<:ForwardMode}, x)
    B = pick_batchsize(backend, length(x))
    shadow = chunkedonehot(x, Val(B))
    return EnzymeForwardGradientExtras{B,typeof(shadow)}(shadow)
end

function DI.gradient(
    f, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras{B}
) where {B}
    grad_tup = gradient(forward_mode(backend), f, x, Val(B); shadow=extras.shadow)
    return reshape(collect(grad_tup), size(x))
end

function DI.value_and_gradient(
    f, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!(
    f, grad, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras{B}
) where {B}
    grad_tup = gradient(forward_mode(backend), f, x, Val(B); shadow=extras.shadow)
    return copyto!(grad, grad_tup)
end

function DI.value_and_gradient!(
    f, grad, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras{B}
) where {B}
    grad_tup = gradient(forward_mode(backend), f, x, Val(B); shadow=extras.shadow)
    return f(x), copyto!(grad, grad_tup)
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{B,O} <: JacobianExtras
    shadow::O
end

function DI.prepare_jacobian(f, backend::AutoEnzyme{<:Union{ForwardMode,Nothing}}, x)
    B = pick_batchsize(backend, length(x))
    shadow = chunkedonehot(x, Val(B))
    return EnzymeForwardOneArgJacobianExtras{C,typeof(shadow)}(shadow)
end

function DI.jacobian(
    f,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras{B},
) where {B}
    jac_wrongshape = jacobian(forward_mode(backend), f, x, Val(B); shadow=extras.shadow)
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    return reshape(jac_wrongshape, ny, nx)
end

function DI.value_and_jacobian(
    f,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!(
    f,
    jac,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

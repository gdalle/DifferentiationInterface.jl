## Pushforward

struct ForwardEnzymePushforwardExtras{F}
    df::F
end

function DI.prepare_pushforward(f, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx)
    df = make_zero(f)
    return ForwardEnzymePushforwardExtras(df)
end

function DI.value_and_pushforward(
    f,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    extras::ForwardEnzymePushforwardExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    y, new_dy = if backend isa AutoDeferredEnzyme
        autodiff_deferred(forward_mode(backend), Duplicated(f, df), Duplicated, x_and_dx)
    else
        autodiff(forward_mode(backend), Duplicated(f, df), Duplicated, x_and_dx)
    end
    return y, new_dy
end

function DI.pushforward(
    f,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    extras::ForwardEnzymePushforwardExtras,
)
    @compat (; df) = extras
    make_zero!(df)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    new_dy = if backend isa AutoDeferredEnzyme
        only(
            autodiff_deferred(
                forward_mode(backend), Duplicated(f, df), DuplicatedNoNeed, x_and_dx
            ),
        )
    else
        only(autodiff(forward_mode(backend), Duplicated(f, df), DuplicatedNoNeed, x_and_dx))
    end
    return new_dy
end

function DI.value_and_pushforward!(
    f,
    dy,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    extras::ForwardEnzymePushforwardExtras,
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
    extras::ForwardEnzymePushforwardExtras,
)
    # dy cannot be passed anyway
    return copyto!(dy, DI.pushforward(f, backend, x, dx, extras))
end

## Gradient

struct EnzymeForwardGradientExtras{B,F,O} <: GradientExtras
    df::F
    shadow::O
end

function DI.prepare_gradient(f, backend::AutoEnzyme{<:ForwardMode}, x)
    B = pick_batchsize(backend, length(x))
    df = make_zero(f)
    shadow = chunkedonehot(x, Val(B))
    return EnzymeForwardGradientExtras{B,typeof(df),typeof(shadow)}(df, shadow)
end

function DI.gradient(
    f, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras{B}
) where {B}
    @compat (; df, shadow) = extras
    make_zero!(df)
    grad_tup = gradient(forward_mode(backend), Duplicated(f, df), x, Val(B); shadow)
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
    @compat (; df, shadow) = extras
    make_zero!(df)
    grad_tup = gradient(forward_mode(backend), Duplicated(f, df), x, Val(B); shadow)
    return copyto!(grad, grad_tup)
end

function DI.value_and_gradient!(
    f, grad, backend::AutoEnzyme{<:ForwardMode}, x, extras::EnzymeForwardGradientExtras{B}
) where {B}
    @compat (; df, shadow) = extras
    make_zero!(df)
    grad_tup = gradient(forward_mode(backend), Duplicated(f, df), x, Val(B); shadow)
    return f(x), copyto!(grad, grad_tup)
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{B,F,O} <: JacobianExtras
    df::F
    shadow::O
end

function DI.prepare_jacobian(f, backend::AutoEnzyme{<:Union{ForwardMode,Nothing}}, x)
    B = pick_batchsize(backend, length(x))
    df = make_zero(f)
    shadow = chunkedonehot(x, Val(B))
    return EnzymeForwardOneArgJacobianExtras{B,typeof(df),typeof(shadow)}(df, shadow)
end

function DI.jacobian(
    f,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras{B},
) where {B}
    @compat (; df, shadow) = extras
    make_zero!(df)
    jac_wrongshape = jacobian(forward_mode(backend), Duplicated(f, df), x, Val(B); shadow)
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

## Pushforward

function DI.prepare_pushforward(
    f, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, tx::Tangents
)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    dy = map(tx.d) do dx
        DI.pushforward(f, backend, x, dx, extras)
    end
    y = f(x)
    return y, Tangents(dy...)
end

function DI.value_and_pushforward(
    f,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
    ::NoPushforwardExtras,
)
    dx = only(tx)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    y, new_dy = if backend isa AutoDeferredEnzyme
        autodiff_deferred(forward_mode(backend), f_and_df, Duplicated, x_and_dx)
    else
        autodiff(forward_mode(backend), f_and_df, Duplicated, x_and_dx)
    end
    return y, Tangents(new_dy)
end

function DI.pushforward(
    f,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
    ::NoPushforwardExtras,
)
    dx = only(tx)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    new_dy = if backend isa AutoDeferredEnzyme
        only(autodiff_deferred(forward_mode(backend), f_and_df, DuplicatedNoNeed, x_and_dx))
    else
        only(autodiff(forward_mode(backend), f_and_df, DuplicatedNoNeed, x_and_dx))
    end
    return Tangents(new_dy)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, backend, x, tx, extras)
    return y, copyto!(ty, new_ty)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    # dy cannot be passed anyway
    return copyto!(ty, DI.pushforward(f, backend, x, tx, extras))
end

## Gradient

struct EnzymeForwardGradientExtras{B,O} <: GradientExtras
    shadow::O
end

function DI.prepare_gradient(
    f, backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}}, x
)
    B = pick_batchsize(backend, length(x))
    shadow = chunkedonehot(x, Val(B))
    return EnzymeForwardGradientExtras{B,typeof(shadow)}(shadow)
end

function DI.gradient(
    f,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardGradientExtras{B},
) where {B}
    f_and_df = get_f_and_df(f, backend)
    grad_tup = gradient(forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow)
    return reshape(collect(grad_tup), size(x))
end

function DI.value_and_gradient(
    f,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardGradientExtras,
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!(
    f,
    grad,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardGradientExtras{B},
) where {B}
    f_and_df = get_f_and_df(f, backend)
    grad_tup = gradient(forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow)
    return copyto!(grad, grad_tup)
end

function DI.value_and_gradient!(
    f,
    grad,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardGradientExtras{B},
) where {B}
    f_and_df = get_f_and_df(f, backend)
    grad_tup = gradient(forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow)
    return f(x), copyto!(grad, grad_tup)
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{B,O} <: JacobianExtras
    shadow::O
end

function DI.prepare_jacobian(
    f, backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}}, x
)
    B = pick_batchsize(backend, length(x))
    if B == 1
        shadow = onehot(x)
    else
        shadow = chunkedonehot(x, Val(B))
    end
    return EnzymeForwardOneArgJacobianExtras{B,typeof(shadow)}(shadow)
end

function DI.jacobian(
    f,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras{B},
) where {B}
    f_and_df = get_f_and_df(f, backend)
    jac_wrongshape = jacobian(
        forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow
    )
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    return reshape(jac_wrongshape, ny, nx)
end

function DI.value_and_jacobian(
    f,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!(
    f,
    jac,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

function DI.value_and_jacobian!(
    f,
    jac,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    extras::EnzymeForwardOneArgJacobianExtras,
)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

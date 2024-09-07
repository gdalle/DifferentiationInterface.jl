## Pushforward

function DI.prepare_pushforward(
    f, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, tx::Tangents
)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f,
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    dys = map(tx.d) do dx
        DI.pushforward(f, extras, backend, x, dx)
    end
    y = f(x)
    return y, Tangents(dys...)
end

function DI.value_and_pushforward(
    f,
    ::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
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
    ::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
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
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, extras, backend, x, tx)
    return y, copyto!(ty, new_ty)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    # dy cannot be passed anyway
    return copyto!(ty, DI.pushforward(f, extras, backend, x, tx))
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
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    grad_tup = gradient(forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow)
    return reshape(collect(grad_tup), size(x))
end

function DI.value_and_gradient(
    f,
    extras::EnzymeForwardGradientExtras,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
)
    return f(x), DI.gradient(f, extras, backend, x)
end

function DI.gradient!(
    f,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    grad_tup = gradient(forward_mode(backend), f_and_df, x, Val(B); shadow=extras.shadow)
    return copyto!(grad, grad_tup)
end

function DI.value_and_gradient!(
    f,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
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
    extras::EnzymeForwardOneArgJacobianExtras{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
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
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return f(x), DI.jacobian(f, extras, backend, x)
end

function DI.jacobian!(
    f,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end

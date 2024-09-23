## Pushforward

function DI.prepare_pushforward(
    f!, y, ::AutoEnzyme{<:Union{ForwardMode,Nothing}}, x, tx::Tangents
)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f!,
    y,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
)
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = convert(typeof(x), only(tx))
    dy_sametype = make_zero(y)
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    autodiff(forward_mode_noprimal(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    return y, Tangents(dy_sametype)
end

function DI.value_and_pushforward(
    f!,
    y,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{B},
) where {B}
    f!_and_df! = get_f_and_df(f!, backend)
    dxs_sametype = map(Fix1(convert, typeof(x)), tx.d)
    dys_sametype = ntuple(_ -> make_zero(y), Val(B))
    x_and_dxs = BatchDuplicated(x, dxs_sametype)
    y_and_dys = BatchDuplicated(y, dys_sametype)
    autodiff(forward_mode_noprimal(backend), f!_and_df!, Const, y_and_dys, x_and_dxs)
    return y, Tangents(dys_sametype...)
end

## Pushforward
struct GTPSATwoArgPushforwardExtras{X,Y} <: PushforwardExtras
    xt::X
    yt::Y
end

function DI.prepare_pushforward(f!, y, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1)
    end

    dx = first(tx)
    if x isa Number
        xt = TPS{promote_type(typeof(dx), typeof(x), Float64)}(; use=d)
    else
        xt = similar(x, TPS{promote_type(eltype(dx), eltype(x), Float64)})

        # xt and x have same indexing because of similar
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(dx), eltype(x), Float64)}(; use=d)
        end
    end

    yt = similar(y, TPS{promote_type(eltype(y), Float64)})

    for i in eachindex(yt)
        yt[i] = TPS{promote_type(eltype(y), Float64)}(; use=d)
    end
    return GTPSATwoArgPushforwardExtras(xt, yt)
end

function DI.pushforward(
    f!, y, extras::GTPSATwoArgPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    ty = map(tx) do dx
        if x isa Number
            extras.xt[0] = x
            extras.xt[1] = dx
        else
            j = 1
            for i in eachindex(x)
                extras.xt[i][0] = x[i]
                extras.xt[i][j] = dx[i]
                j += 1
            end
        end

        f!(extras.yt, extras.xt)
        map!(t -> t[0], y, extras.yt)
        dy = similar(extras.yt, eltype(eltype(extras.yt)))
        dy .= 0
        for i in eachindex(extras.yt)
            for j in 1:length(dx)
                dy[i] += extras.yt[i][j]
            end
        end
        return dy
    end

    return ty
end

function DI.pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::GTPSATwoArgPushforwardExtras,
    backend::AutoGTPSA,
    x,
    tx::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        if x isa Number
            extras.xt[0] = x
            extras.xt[1] = dx
        else
            j = 1
            for i in eachindex(x)
                extras.xt[i][0] = x[i]
                extras.xt[i][j] = dx[i]
                j += 1
            end
        end

        f!(extras.yt, extras.xt)
        map!(t -> t[0], y, extras.yt)
        dy .= 0
        for i in eachindex(extras.yt)
            for j in 1:length(dx)
                dy[i] += extras.yt[i][j]
            end
        end
    end

    return ty
end

function DI.value_and_pushforward(
    f!, y, extras::GTPSATwoArgPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    ys_and_dys = map(tx.d) do dx
        if x isa Number
            extras.xt[0] = x
            extras.xt[1] = dx
        else
            j = 1
            for i in eachindex(x)
                extras.xt[i][0] = x[i]
                extras.xt[i][j] = dx[i]
                j += 1
            end
        end

        f!(extras.yt, extras.xt)
        dy = similar(extras.yt, eltype(eltype(extras.yt)))
        dy .= 0
        for i in eachindex(extras.yt)
            for j in 1:length(dx)
                dy[i] += extras.yt[i][j]
            end
        end
        map!(t -> t[0], y, extras.yt)

        return y, dy
    end

    y = first(ys_and_dys[1])
    dys = last.(ys_and_dys)
    ty = Tangents(dys...)
    return y, ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::GTPSATwoArgPushforwardExtras,
    backend::AutoGTPSA,
    x,
    tx::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        if x isa Number
            extras.xt[0] = x
            extras.xt[1] = dx
        else
            j = 1
            for i in eachindex(x)
                extras.xt[i][0] = x[i]
                extras.xt[i][j] = dx[i]
                j += 1
            end
        end

        f!(extras.yt, extras.xt)
        map!(t -> t[0], y, extras.yt)
        dy .= 0
        for i in eachindex(extras.yt)
            for j in 1:length(dx)
                dy[i] += extras.yt[i][j]
            end
        end
    end

    return y, ty
end

## Jacobian

struct GTPSATwoArgJacobianExtras{X,Y} <: JacobianExtras
    xt::X
    yt::Y
end

function DI.prepare_jacobian(f!, y, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
    else
        nn = length(x)
        d = Descriptor(nn, 1)
    end
    xt = similar(x, TPS{promote_type(eltype(x), Float64)})

    # xt and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x), Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    yt = similar(y, TPS{promote_type(eltype(y), Float64)})

    for i in eachindex(yt)
        yt[i] = TPS{promote_type(eltype(y), Float64)}(; use=d)
    end

    return GTPSATwoArgJacobianExtras(xt, yt)
end

function DI.jacobian(f!, y, extras::GTPSATwoArgJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    jac = similar(x, eltype(eltype(extras.yt)), (length(extras.yt), length(x)))
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    map!(t -> t[0], y, extras.yt)
    return jac
end

function DI.jacobian!(f!, y, jac, extras::GTPSATwoArgJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    map!(t -> t[0], y, extras.yt)
    return jac
end

function DI.value_and_jacobian(f!, y, extras::GTPSATwoArgJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    jac = similar(x, eltype(eltype(extras.yt)), (length(extras.yt), length(x)))
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    map!(t -> t[0], y, extras.yt)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, extras::GTPSATwoArgJacobianExtras, ::AutoGTPSA, x
)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    map!(t -> t[0], y, extras.yt)
    return y, jac
end

## Pushforward
struct GTPSATwoArgPushforwardExtras{X,Y} <: PushforwardExtras
    xt::X
    yt::Y
end

function DI.prepare_pushforward(f!, y, backend::AutoGTPSA{D}, x, dx) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1)
    end

    if x isa Number
        xt = TPS{promote_type(typeof(dx),typeof(x),Float64)}(; use=d)
    else
        xt = similar(x, TPS{promote_type(eltype(dx),eltype(x),Float64)})

        # xt and x have same indexing because of similar
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(dx),eltype(x),Float64)}(; use=d)
        end
    end

    if y isa Number
        yt = TPS{promote_type(typeof(y),Float64)}(; use=d)
    else
        yt = similar(y, TPS{promote_type(eltype(y),Float64)})

        for i in eachindex(yt)
            yt[i] = TPS{promote_type(eltype(y),Float64)}(; use=d)
        end
    end

    return GTPSATwoArgPushforwardExtras(xt, yt)
end

function DI.pushforward(f!, y, backend::AutoGTPSA, x, dx, extras::GTPSATwoArgPushforwardExtras)
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
    if extras.yt isa Number
        return extras.yt[1]
    else
        dy = similar(extras.yt, eltype(eltype(extras.yt)))
        j = 1
        for i in eachindex(extras.yt)
            dy[i] = extras.yt[i][j]
            j += 1
        end
        return dy
    end
end

function DI.pushforward!(f!, y, dy, backend::AutoGTPSA, x, dx, extras::GTPSATwoArgPushforwardExtras)
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
    if extras.yt isa Number
        return extras.yt[1]
    else
        j = 1
        for i in eachindex(extras.yt)
            dy[i] = extras.yt[i][j]
            j += 1
        end
        return dy
    end
end

function DI.value_and_pushforward(f!, y, backend::AutoGTPSA, x, dx, extras::GTPSATwoArgPushforwardExtras)
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
    if extras.yt isa Number
        return extras.yt[1]
    else
        dy = similar(extras.yt, eltype(eltype(extras.yt)))
        j = 1
        for i in eachindex(extras.yt)
            dy[i] = extras.yt[i][j]
            j += 1
        end
        y = map(t->t[0], extras.yt)
        return y, dy
    end
end

function DI.value_and_pushforward!(f!, y, dy, backend::AutoGTPSA, x, dx, extras::GTPSATwoArgPushforwardExtras)
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
    if extras.yt isa Number
        return extras.yt[1]
    else
        j = 1
        for i in eachindex(extras.yt)
            dy[i] = extras.yt[i][j]
            j += 1
        end
        y = map(t->t[0], extras.yt)
        return y, dy
    end
end

## Derivative

struct GTPSATwoArgDerivativeExtras{X,Y} <: TwoArgDerivativeExtras
    xt::X
    yt::Y
end

function DI.prepare_derivative(f!, y, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 1)
    end
    xt = TPS{promote_type(typeof(x),Float64)}(; use=d)
    xt[1] = 1

    if y isa Number
        yt = TPS{promote_type(typeof(y),Float64)}(; use=d)
    else
        yt = similar(y, TPS{promote_type(eltype(y),Float64)})

        for i in eachindex(yt)
            yt[i] = TPS{promote_type(eltype(y),Float64)}(; use=d)
        end
    end

    return GTPSATwoArgDerivativeExtras(xt, yt)
end

function DI.derivative(f!, y, ::AutoGTPSA, x, extras::GTPSATwoArgDerivativeExtras)
    extras.xt[0] = x
    f!(extras.yt, extras.xt)
    if extras.yt isa Number
        return extras.yt[1]
    else
        der = similar(extras.yt, eltype(eltype(extras.yt)))
        for i in eachindex(extras.yt)
            der[i] = extras.yt[i][1]
        end
        return der
    end
end

function DI.derivative!(f!, y, der, ::AutoGTPSA, x, extras::GTPSATwoArgDerivativeExtras)
    extras.xt[0] = x
    f!(extras.yt, extras.xt)
    for i in eachindex(extras.yt)
        der[i] = extras.yt[i][1]
    end
    return der
end

function DI.value_and_derivative(f!, y, ::AutoGTPSA, x, extras::GTPSATwoArgDerivativeExtras)
    extras.xt[0] = x
    f!(extras.yt, extras.xt)
    if extras.yt isa Number
        return extras.yt[0], extras.yt[1]
    else
        y = map(t -> t[0], extras.yt)
        der = similar(extras.yt, eltype(eltype(extras.yt)))
        for i in eachindex(extras.yt)
            der[i] = extras.yt[i][1]
        end
        return y, der
    end
end

function DI.value_and_derivative!(f!, y, der, ::AutoGTPSA, x, extras::GTPSATwoArgDerivativeExtras)
    extras.xt[0] = x
    f!(extras.yt, extras.xt)
    y = map(t -> t[0], extras.yt)
    for i in eachindex(extras.yt)
        der[i] = extras.yt[i][1]
    end
    return y, der
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
    xt = similar(x, TPS{promote_type(eltype(x),Float64)})

    # xt and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x),Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    if y isa Number
        yt = TPS{promote_type(typeof(y),Float64)}(; use=d)
    else
        yt = similar(y, TPS{promote_type(eltype(y),Float64)})

        for i in eachindex(yt)
            yt[i] = TPS{promote_type(eltype(y),Float64)}(; use=d)
        end
    end

    return GTPSATwoArgJacobianExtras(xt, yt)
end

function DI.jacobian(f!, y, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    jac = similar(x, eltype(eltype(extras.yt)), (length(extras.yt), length(x)))
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    return jac
end

function DI.jacobian!(f!, y, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    return jac
end

function DI.value_and_jacobian(f!, y, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    jac = similar(x, eltype(eltype(extras.yt)), (length(extras.yt), length(x)))
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    y = map(t -> t[0], extras.yt)
    return y, jac
end

function DI.value_and_jacobian!(f!, y, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    f!(extras.yt, extras.xt)
    GTPSA.jacobian!(jac, extras.yt; include_params=true)
    y = map(t -> t[0], extras.yt)
    return y, jac
end

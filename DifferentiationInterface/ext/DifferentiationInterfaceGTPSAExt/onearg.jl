## Pushforward
struct GTPSAPushforwardExtras{X} <: PushforwardExtras
    xt::X
end

function DI.prepare_pushforward(f, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1)
    end

    dx = first(tx)
    if x isa Number
        xt = TPS{promote_type(typeof(dx), typeof(x), Float64)}(; use=d)
        return GTPSAPushforwardExtras(xt)
    else
        xt = similar(x, TPS{promote_type(eltype(dx), eltype(x), Float64)})

        # xt and x have same indexing because of similar
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(dx), eltype(x), Float64)}(; use=d)
        end
        return GTPSAPushforwardExtras(xt)
    end
end

function DI.pushforward(
    f, extras::GTPSAPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    ty = map(tx) do dx
        initialize!(extras.xt, x, dx)

        yt = f(extras.xt)
        if yt isa Number
            if dx isa Number
                return yt[1]
            else
                dy = 0
                for j in 1:length(dx)
                    dy += yt[j]
                end

                return dy
            end
        else
            dy = similar(yt, eltype(eltype(yt)))
            dy .= 0
            for i in eachindex(yt)
                for j in 1:length(dx)
                    dy[i] += yt[i][j]
                end
            end

            return dy
        end
    end
    return ty
end

function DI.pushforward!(
    f, ty::Tangents, extras::GTPSAPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        initialize!(extras.xt, x, dx)

        yt = f(extras.xt)
        dy .= 0
        for i in eachindex(yt)
            for j in 1:length(dx)
                dy[i] += yt[i][j]
            end
        end
    end
    return ty
end

function DI.value_and_pushforward(
    f, extras::GTPSAPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    ys_and_dys = map(tx.d) do dx
        initialize!(extras.xt, x, dx)

        yt = f(extras.xt)
        if yt isa Number
            if dx isa Number
                return yt[0], yt[1]
            else
                dy = 0
                for j in 1:length(dx)
                    dy += yt[j]
                end

                return yt[0], dy
            end
        else
            dy = similar(yt, eltype(eltype(yt)))
            dy .= 0
            for i in eachindex(yt)
                for j in 1:length(dx)
                    dy[i] += yt[i][j]
                end
            end
            y = map(t -> t[0], yt)

            return y, dy
        end
    end
    y = first(ys_and_dys[1])
    dys = last.(ys_and_dys)
    ty = Tangents(dys...)
    return y, ty
end

function DI.value_and_pushforward!(
    f, ty::Tangents, extras::GTPSAPushforwardExtras, backend::AutoGTPSA, x, tx::Tangents
)
    for b in eachindex(tx.d, ty.d)
        dx, dy = tx.d[b], ty.d[b]
        initialize!(extras.xt, x, dx)

        yt = f(extras.xt)
        dy .= 0
        for i in eachindex(yt)
            for j in 1:length(dx)
                dy[i] += yt[i][j]
            end
        end
    end
    y = f(x)  # TODO: optimize
    return y, ty
end

## Gradient

struct GTPSAGradientExtras{X} <: GradientExtras
    xt::X
end

function DI.prepare_gradient(f, backend::AutoGTPSA{D}, x) where {D}
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

    return GTPSAGradientExtras(xt)
end

function DI.gradient(f, extras::GTPSAGradientExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.gradient!(f, grad, extras::GTPSAGradientExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.value_and_gradient(f, extras::GTPSAGradientExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

function DI.value_and_gradient!(f, grad, extras::GTPSAGradientExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

## Jacobian

struct GTPSAJacobianExtras{X} <: JacobianExtras
    xt::X
end

function DI.prepare_jacobian(f, backend::AutoGTPSA{D}, x) where {D}
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

    return GTPSAJacobianExtras(xt)
end

function DI.jacobian(f, extras::GTPSAJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.jacobian!(f, jac, extras::GTPSAJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.value_and_jacobian(f, extras::GTPSAJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

function DI.value_and_jacobian!(f, jac, extras::GTPSAJacobianExtras, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

## Second derivative

struct GTPSASecondDerivativeExtras{X} <: SecondDerivativeExtras
    xt::X
end

function DI.prepare_second_derivative(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 2)
    end
    xt = TPS{promote_type(typeof(x), Float64)}(; use=d)
    xt[1] = 1
    return GTPSASecondDerivativeExtras(xt)
end

function DI.second_derivative(f, extras::GTPSASecondDerivativeExtras, ::AutoGTPSA, x)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[2] * 2
    else
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der2[i] = yt[i][2] * 2 # *2 because monomial coefficient is 1/2
        end
        return der2
    end
end

function DI.second_derivative!(f, der2, extras::GTPSASecondDerivativeExtras, ::AutoGTPSA, x)
    extras.xt[0] = x
    yt = f(extras.xt)
    for i in eachindex(yt)
        der2[i] = yt[i][2] * 2
    end
    return der2
end

function DI.value_derivative_and_second_derivative(
    f, extras::GTPSASecondDerivativeExtras, ::AutoGTPSA, x
)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[0], yt[1], yt[2] * 2
    else
        y = map(t -> t[0], yt)
        der = similar(yt, eltype(eltype(yt)))
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
            der2[i] = yt[i][2] * 2
        end
        return y, der, der2
    end
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, extras::GTPSASecondDerivativeExtras, ::AutoGTPSA, x
)
    extras.xt[0] = x
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
        der2[i] = yt[i][2] * 2
    end
    return y, der, der2
end

## Hessian

struct GTPSAHessianExtras{X,M} <: HessianExtras
    xt::X
    m::M
end

function DI.prepare_hessian(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
        m = Vector{UInt8}(undef, nn)
    else
        nn = length(x)
        d = Descriptor(nn, 2)
        # If all variables/variable+parameters have truncation order > 2, then 
        # the indexing is known beforehand and we can do it (very slightly) faster
        m = nothing
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

    return GTPSAHessianExtras(xt, m)
end

function DI.hessian(f, extras::GTPSAHessianExtras, ::AutoGTPSA{D}, x) where {D}
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=extras.m
    )
    return hess
end

function DI.hessian!(f, hess, extras::GTPSAHessianExtras, ::AutoGTPSA{D}, x) where {D}
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=extras.m
    )
    return hess
end

function DI.value_gradient_and_hessian(
    f, extras::GTPSAHessianExtras, ::AutoGTPSA{D}, x
) where {D}
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=extras.m
    )
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, extras::GTPSAHessianExtras, ::AutoGTPSA{D}, x
) where {D}
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    GTPSA.gradient!(grad, yt; include_params=true)
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=extras.m
    )
    return y, grad, hess
end

# HVP

struct GTPSAHVPExtras{E,H} <: HVPExtras
    hessextras::E
    hess::H
end

function DI.prepare_hvp(f, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    hessextras = DI.prepare_hessian(f, backend, x)
    hess = similar(x, typeof(f(x)), (length(x), length(x)))
    return GTPSAHVPExtras(hessextras, hess)
end

function DI.hvp(f, extras::GTPSAHVPExtras, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    DI.hessian!(f, extras.hess, extras.hessextras, backend, x)
    tg = map(tx) do dx
        dg = similar(x, eltype(extras.hess))
        dg .= 0
        j = 1
        for dxi in dx
            for i in 1:size(extras.hess, 2)
                dg[i] += extras.hess[i, j] * dxi
            end
            j += 1
        end
        return dg
    end
    return tg
end

function DI.hvp!(
    f, tg::Tangents, extras::GTPSAHVPExtras, backend::AutoGTPSA{D}, x, tx::Tangents
) where {D}
    DI.hessian!(f, extras.hess, extras.hessextras, backend, x)
    for b in eachindex(tg.d)
        dx, dg = tx.d[b], tg.d[b]
        dg .= 0
        j = 1
        for dxi in dx
            for i in 1:size(extras.hess, 2)
                dg[i] += extras.hess[i, j] * dxi
            end
            j += 1
        end
    end
    return tg
end

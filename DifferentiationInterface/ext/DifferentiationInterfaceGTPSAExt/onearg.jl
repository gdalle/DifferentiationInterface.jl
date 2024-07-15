## Pushforward
struct GTPSAPushforwardExtras{X} <: PushforwardExtras
    xd::X
end

function DI.prepare_pushforward(f, backend::AutoGTPSA{D}, x, dx) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1)
    end

    if x isa Number
        t = TPS(; use=d)
        return GTPSAPushforwardExtras(t)
    else
        v = similar(x, TPS)

        # v and x have same indexing because of similar
        for i in eachindex(v)
            v[i] = TPS(; use=d)
        end
        return GTPSAPushforwardExtras(v)
    end
end

function DI.pushforward(f, backend::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras)
    if x isa Number
        extras.xd[0] = x
        extras.xd[1] = dx
    else
        j = 1
        for i in eachindex(x)
            extras.xd[i][0] = x[i]
            extras.xd[i][j] = dx[i]
            j += 1
        end
    end

    yt = f(extras.xd)
    if yt isa Number
        return yt[1]
    else
        dy = similar(yt, GTPSA.numtype(eltype(yt)))
        j = 1
        for i in eachindex(yt)
            dy[i] = yt[i][j]
            j += 1
        end
        return dy
    end
end

function DI.pushforward!(f, dy, backend::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras)
    if x isa Number
        extras.xd[0] = x
        extras.xd[1] = dx
    else
        j = 1
        for i in eachindex(x)
            extras.xd[i][0] = x[i]
            extras.xd[i][j] = dx[i]
            j += 1
        end
    end

    yt = f(extras.xd)
    if yt isa Number
        return yt[1]
    else
        j = 1
        for i in eachindex(yt)
            dy[i] = yt[i][j]
            j += 1
        end
        return dy
    end
end

function DI.value_and_pushforward(f, backend::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras)
    if x isa Number
        extras.xd[0] = x
        extras.xd[1] = dx
    else
        j = 1
        for i in eachindex(x)
            extras.xd[i][0] = x[i]
            extras.xd[i][j] = dx[i]
            j += 1
        end
    end

    yt = f(extras.xd)
    if yt isa Number
        return yt[1]
    else
        dy = similar(yt, GTPSA.numtype(eltype(yt)))
        j = 1
        for i in eachindex(yt)
            dy[i] = yt[i][j]
            j += 1
        end
        y = map(t->t[0], yt)
        return y, dy
    end
end

function DI.value_and_pushforward!(f, dy, backend::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras)
    if x isa Number
        extras.xd[0] = x
        extras.xd[1] = dx
    else
        j = 1
        for i in eachindex(x)
            extras.xd[i][0] = x[i]
            extras.xd[i][j] = dx[i]
            j += 1
        end
    end

    yt = f(extras.xd)
    if yt isa Number
        return yt[1]
    else
        j = 1
        for i in eachindex(yt)
            dy[i] = yt[i][j]
            j += 1
        end
        y = map(t->t[0], yt)
        return y, dy
    end
end

## Derivative

struct GTPSADerivativeExtras{T} <: DerivativeExtras
    t::T
end

function DI.prepare_derivative(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 1)
    end
    t = TPS(; use=d)
    t[1] = 1
    return GTPSADerivativeExtras(t)
end

function DI.derivative(f, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    if yt isa Number
        return yt[1]
    else
        der = similar(yt, GTPSA.numtype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
        end
        return der
    end
end

function DI.derivative!(f, der, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    for i in eachindex(yt)
        der[i] = yt[i][1]
    end
    return der
end

function DI.value_and_derivative(f, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    if yt isa Number
        return yt[0], yt[1]
    else
        y = map(t -> t[0], yt)
        der = similar(yt, GTPSA.numtype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
        end
        return y, der
    end
end

function DI.value_and_derivative!(f, der, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
    end
    return y, der
end

## Gradient

struct GTPSAGradientExtras{V} <: GradientExtras
    v::V
end

function DI.prepare_gradient(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
    else
        nn = length(x)
        d = Descriptor(nn, 1)
    end
    v = similar(x, TPS)

    # v and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(v)
        v[i] = TPS(; use=d)
        v[i][j] = 1
        j += 1
    end

    return GTPSAGradientExtras(v)
end

function DI.gradient(f, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    grad = similar(x, GTPSA.numtype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.gradient!(f, grad, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.value_and_gradient(f, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    grad = similar(x, GTPSA.numtype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

function DI.value_and_gradient!(f, grad, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

## Jacobian

struct GTPSAJacobianExtras{V} <: JacobianExtras
    v::V
end

function DI.prepare_jacobian(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
    else
        nn = length(x)
        d = Descriptor(nn, 1)
    end
    v = similar(x, TPS)

    # v and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(v)
        v[i] = TPS(; use=d)
        v[i][j] = 1
        j += 1
    end

    return GTPSAJacobianExtras(v)
end

function DI.jacobian(f, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    jac = similar(x, GTPSA.numtype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.jacobian!(f, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.value_and_jacobian(f, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    jac = similar(x, GTPSA.numtype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

function DI.value_and_jacobian!(f, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

## Second derivative

struct GTPSASecondDerivativeExtras{T} <: SecondDerivativeExtras
    t::T
end

function DI.prepare_second_derivative(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 2)
    end
    t = TPS(; use=d)
    t[1] = 1
    return GTPSASecondDerivativeExtras(t)
end

function DI.second_derivative(f, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    if yt isa Number
        return yt[2]
    else
        der2 = similar(yt, GTPSA.numtype(eltype(yt)))
        for i in eachindex(yt)
            der2[i] = yt[i][2]
        end
        return der2
    end
end

function DI.second_derivative!(f, der2, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras)
    extras.t[0] = x
    yt = f(extras.t)
    for i in eachindex(yt)
        der2[i] = yt[i][2]
    end
    return der2
end

function DI.value_derivative_and_second_derivative(
    f, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras
)
    extras.t[0] = x
    yt = f(extras.t)
    if yt isa Number
        return yt[0], yt[1], yt[2]
    else
        y = map(t -> t[0], yt)
        der = similar(yt, GTPSA.numtype(eltype(yt)))
        der2 = similar(yt, GTPSA.numtype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
            der2[i] = yt[i][2]
        end
        return y, der, der2
    end
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras
)
    extras.t[0] = x
    yt = f(extras.t)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
        der2[i] = yt[i][2]
    end
    return y, der, der2
end

## Hessian

struct GTPSAHessianExtras{V} <: HessianExtras
    v::V
end

function DI.prepare_hessian(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
    else
        nn = length(x)
        d = Descriptor(nn, 2)
    end
    v = similar(x, TPS)

    # v and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(v)
        v[i] = TPS(; use=d)
        v[i][j] = 1
        j += 1
    end

    return GTPSAHessianExtras(v)
end

function DI.hessian(f, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    hess = similar(x, GTPSA.numtype(eltype(yt)), (length(x), length(x)))
    GTPSA.hessian!(hess, yt; include_params=true)
    return hess
end

function DI.hessian!(f, hess, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    GTPSA.hessian!(hess, yt; include_params=true)
    return hess
end

function DI.value_gradient_and_hessian(f, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    y = map(t -> t[0], yt)
    grad = similar(x, GTPSA.numtype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    hess = similar(x, GTPSA.numtype(eltype(yt)), (length(x), length(x)))
    GTPSA.hessian!(hess, yt; include_params=true)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, ::AutoGTPSA, x, extras::GTPSAHessianExtras
)
    foreach((t, xi) -> t[0] = xi, extras.v, x) # Set the scalar part
    yt = f(extras.v)
    y = map(t -> t[0], yt)
    GTPSA.gradient!(grad, yt; include_params=true)
    GTPSA.hessian!(hess, yt; include_params=true)
    return y, grad, hess
end

## Pushforward
struct GTPSAPushforwardExtras{X} <: PushforwardExtras
    xt::X
end

function DI.prepare_pushforward(f, backend::AutoGTPSA{D}, x, dx) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1)
    end

    if x isa Number
        xt = TPS{promote_type(typeof(dx),typeof(x),Float64)}(; use=d)
        return GTPSAPushforwardExtras(xt)
    else
        xt = similar(x, TPS{promote_type(eltype(dx),eltype(x),Float64)})

        # xt and x have same indexing because of similar
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(dx),eltype(x),Float64)}(; use=d)
        end
        return GTPSAPushforwardExtras(xt)
    end
end

function DI.pushforward(f, backend::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras)
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

    yt = f(extras.xt)
    if yt isa Number
        return yt[1]
    else
        dy = similar(yt, eltype(eltype(yt)))
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

    yt = f(extras.xt)
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

    yt = f(extras.xt)
    if yt isa Number
        return yt[1]
    else
        dy = similar(yt, eltype(eltype(yt)))
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

    yt = f(extras.xt)
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

struct GTPSADerivativeExtras{X} <: DerivativeExtras
    xt::X
end

function DI.prepare_derivative(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 1)
    end
    xt = TPS{promote_type(typeof(x),Float64)}(; use=d)
    xt[1] = 1
    return GTPSADerivativeExtras(xt)
end

function DI.derivative(f, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[1]
    else
        der = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
        end
        return der
    end
end

function DI.derivative!(f, der, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
    end
    return der
end

function DI.value_and_derivative(f, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[0], yt[1]
    else
        y = map(t -> t[0], yt)
        der = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
        end
        return y, der
    end
end

function DI.value_and_derivative!(f, der, ::AutoGTPSA, x, extras::GTPSADerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
    end
    return y, der
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

    return GTPSAGradientExtras(xt)
end

function DI.gradient(f, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.gradient!(f, grad, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

function DI.value_and_gradient(f, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

function DI.value_and_gradient!(f, grad, ::AutoGTPSA, x, extras::GTPSAGradientExtras)
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

    return GTPSAJacobianExtras(xt)
end

function DI.jacobian(f, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.jacobian!(f, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.value_and_jacobian(f, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

function DI.value_and_jacobian!(f, jac, ::AutoGTPSA, x, extras::GTPSAJacobianExtras)
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
    xt = TPS{promote_type(typeof(x),Float64)}(; use=d)
    xt[1] = 1
    return GTPSASecondDerivativeExtras(xt)
end

function DI.second_derivative(f, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[2]
    else
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der2[i] = yt[i][2]/2
        end
        return der2
    end
end

function DI.second_derivative!(f, der2, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras)
    extras.xt[0] = x
    yt = f(extras.xt)
    for i in eachindex(yt)
        der2[i] = yt[i][2]/2
    end
    return der2
end

function DI.value_derivative_and_second_derivative(
    f, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras
)
    extras.xt[0] = x
    yt = f(extras.xt)
    if yt isa Number
        return yt[0], yt[1], yt[2]
    else
        y = map(t -> t[0], yt)
        der = similar(yt, eltype(eltype(yt)))
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
            der2[i] = yt[i][2]/2
        end
        return y, der, der2
    end
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, ::AutoGTPSA, x, extras::GTPSASecondDerivativeExtras
)
    extras.xt[0] = x
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
        der2[i] = yt[i][2]/2
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

    return GTPSAHessianExtras(xt, m)
end

function DI.hessian(f, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    GTPSA.hessian!(hess, yt; include_params=true)
    return hess
end

function DI.hessian!(f, hess, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    GTPSA.hessian!(hess, yt; include_params=true)
    return hess
end

function DI.value_gradient_and_hessian(f, ::AutoGTPSA, x, extras::GTPSAHessianExtras)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    GTPSA.hessian!(hess, yt; include_params=true)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, ::AutoGTPSA, x, extras::GTPSAHessianExtras
)
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    yt = f(extras.xt)
    y = map(t -> t[0], yt)
    GTPSA.gradient!(grad, yt; include_params=true)
    GTPSA.hessian!(hess, yt; include_params=true)
    return y, grad, hess
end

# HVP
struct GTPSAHVPExtras{X,M} <: HVPExtras
    xt::X
    m::M
end

function DI.prepare_hvp(f, backend::AutoGTPSA{D}, x, dx) where {D}
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
    xt = similar(x, TPS{promote_type(eltype(dx),eltype(x),Float64)})

    # xt and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(dx),eltype(x),Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    return GTPSAHVPExtras(xt, mono) 
end

function DI.hvp(f, backend::AutoGTPSA{D}, x, dx, extras::GTPSAHVPExtras) where {D}
    foreach((t, xi) -> t[0] = xi, extras.xt, x) # Set the scalar part
    
    yt = f(extras.xt)
    

    dg = similar(x, eltype(eltype(yt)))
    dg .= 0

    d = GTPSA.getdesc(yt) 
    desc = unsafe_load(d.desc)
    nn = desc.nn


    if D == Nothing
        idx = desc.nv+desc.np
        endidx = floor(nn*(nn+1)/2)+nn
        curdiag = 1
        col = 1
        xt = Ref{eltype(yt)}()
        idx = cycle!(yt, idx, 0, C_NULL, xt)
        while idx <= endidx && idx > 0
            h_idx = idx-nn
            while h_idx > curdiag
            col += 1
            curdiag += col
            end
            row = col-(curdiag-h_idx)
            #println("row = ", row, ", col = ", col)
            if row==col
                dg[row] += 2*xt[]*dx[col]
            else
                dg[row] += xt[]*dx[col]
                dg[col] += xt[]*dx[row]
            end
            idx = cycle!(yt, idx, 0, C_NULL, xt)
        end
    else
      # If there are some variables/parameters with TO == 1, we have to do it "slow"
      # because the indexing of TPSA index -> hessian index can be very complicated.
      # I saw slow in quotes because it is likely still much faster than the calculation
      # of the Hessian itself (this is just a getter)  
      idx = desc.nv+desc.np # start at 2nd order
      xt = Ref{eltype(yt)}()
      mono = extras.mono
      idx = cycle!(yt, idx, nn, mono, xt)
      while idx > 0 
        if sum(mono) > 0x2
          return dg
        end
        i = findfirst(x->x==0x1, mono)
        if isnothing(i)
          i = findfirst(x->x==0x2, mono)
          if isnothing(i)
            return dg
          end
          if i <= nn
            dg[i] += 2*xt[]*dx[i]   # Multiply by 2 because taylor coefficient on diagonal is 1/2!*d2f/dx2
          end
        else 
          j = findlast(x->x==0x1, mono)
          if isnothing(j)
            return dg
          end
          if i <= nn && j <= nn
            dg[i] += xt[]*dx[j]
            dg[j] += xt[]*dx[i]
          end
        end
        idx = cycle!(yt, idx, nn, mono, xt)
      end
    end
    return dg
end
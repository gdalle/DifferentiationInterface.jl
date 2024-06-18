# --- derivative ---
struct GTPSADerivativeExtras <: DerivativeExtras
  t::TPS
end

function DI.prepare_derivative(f, backend::AutoGTPSA{D}, x) where {D}
  if D != Nothing
    d = backend.descriptor
  else
    d = Descriptor(1,1)
  end
  t = TPS(use=d)
  t[1] = 1
  return GTPSADerivativeExtras(t)
end

function DI.derivative(f, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
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

function DI.derivative!(f, der, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number # This should never be reached
    return yt[1]
  else
    for i in eachindex(yt)
      der[i] = yt[i][1]
    end
    return der
  end
end

function DI.value_and_derivative(f, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number
    return yt[0], yt[1]
  else
    y = map(t->t[0], yt)
    der = similar(yt, GTPSA.numtype(eltype(yt)))
    for i in eachindex(yt)
      der[i] = yt[i][1]
    end
    return y, der
  end
end

function DI.value_and_derivative!(f, der, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number # This should never be reached
    return yt[0], yt[1]
  else
    y = map(t->t[0], yt)
    for i in eachindex(yt)
      der[i] = yt[i][1]
    end
    return y, der
  end
end

# --- second_derivative ---
struct GTPSASecondDerivativeExtras <: SecondDerivativeExtras
  t::TPS
end

function DI.prepare_second_derivative(f, backend::AutoGTPSA{D}, x) where {D}
  if D != Nothing
    d = backend.descriptor
  else
    d = Descriptor(1,2)
  end
  t = TPS(use=d)
  t[1] = 1
  return GTPSASecondDerivativeExtras(t)
end

function DI.second_derivative(f, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
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

function DI.second_derivative!(f, der2, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number # This should never be reached
    return yt[2]
  else
    for i in eachindex(yt)
      der2[i] = yt[i][2]
    end
    return der2
  end
end

function DI.value_derivative_and_second_derivative(f, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number
    return yt[0], yt[1], yt[2]
  else
    y = map(t->t[0], yt)
    der = similar(yt, GTPSA.numtype(eltype(yt)))
    der2 = similar(yt, GTPSA.numtype(eltype(yt)))
    for i in eachindex(yt)
      der[i] = yt[i][1]
      der2[i] = yt[i][2]
    end
    return y, der, der2
  end
end

function DI.value_derivative_and_second_derivative!(f, der, der2, backend::AutoGTPSA, x, extras::GTPSADerivativeExtras)
  extras.t[0] = x
  yt = f(extras.t)
  if yt isa Number # This should never be reached
    return yt[0], yt[1], yt[2]
  else
    y = map(t->t[0], yt)
    for i in eachindex(yt)
      der[i] = yt[i][1]
      der2[i] = yt[i][2]
    end
    return y, der, der2
  end
end

# --- gradient ---
struct GTPSAGradientExtras <: GradientExtras
  v::Vector{TPS}
end

function DI.prepare_gradient(f, backend::AutoGTPSA{D}, x) where {D}
  if D != Nothing
    d = backend.descriptor
    nn = GTPSA.numnn(d)
  else
    nn = length(x)
    d = Descriptor(nn,1)
  end
  v = similar(x, TPS)

  # v and x have same indexing because of similar
  # Setting the first derivatives must be 1-based 
  # linear with the variables.
  j = 1
  for i in eachindex(v)
    v[i] = TPS(use=d)
    v[i][j] = 1
    j += 1
  end

  return GTPSAGradientExtras(v)
end

function DI.gradient(f, backend::AutoGTPSA, x, extras::GTPSAGradientExtras)
  foreach((t,xi)->t[0]=xi, extras.v, x) # Set the scalar part
  yt = f(extras.v)
  grad = similar(x, GTPSA.numtype(eltype(yt)))
  GTPSA.gradient!(grad, yt, include_params=true)
  return grad
end

function DI.gradient!(f, grad, backend::AutoGTPSA, x, extras::GTPSAGradientExtras)
  foreach((t,xi)->t[0]=xi, extras.v, x) # Set the scalar part
  yt = f(extras.v)
  GTPSA.gradient!(grad, yt, include_params=true)
  return grad
end

function DI.value_and_gradient(f, backend::AutoGTPSA, x, extras::GTPSAGradientExtras)
  foreach((t,xi)->t[0]=xi, extras.v, x) # Set the scalar part
  yt = f(extras.v)
  grad = similar(x, GTPSA.numtype(eltype(yt)))
  GTPSA.gradient!(grad, yt, include_params=true)
  y = map(t->t[0], yt)
  return y, grad
end

function DI.value_and_gradient!(f, grad, backend::AutoGTPSA, x, extras::GTPSAGradientExtras)
  foreach((t,xi)->t[0]=xi, extras.v, x) # Set the scalar part
  yt = f(extras.v)
  GTPSA.gradient!(grad, yt, include_params=true)
  y = map(t->t[0], yt)
  return y, grad
end

# --- jacobian ---
struct GTPSAJacobianExtras <: JacobianExtras
  v::Vector{TPS}
end




# --- pushforward ---

"""
    GTPSAPushforwardExtras{T} <: PushforwardExtras

This struct contains pre-allocated TPS(s) corresponding to `x` with seed
`dx` for the derivative. If taking derivative of single-variable function, 
then T is a TPS, else T is some vector of TPSs. 
"""
struct GTPSAPushforwardExtras{T} <: PushforwardExtras
  v::T
end

function DI.prepare_pushforward(f, backend::AutoGTPSA, x, dx)
  if x isa Number
    d = Descriptor(1,1)
    t = TPS(use=d)
    t[0] = x
    t[1] = dx
    return GTPSAPushforwardExtras(t)
  else
    NV = length(x)
    d = Descriptor(NV,1) # only first order
    v = similar(x, TPS)

    # v and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(v)
      v[i] = TPS(use=d) # allocate
      v[i][0] = x[i]
      v[i][j] = dx[i]
      j += 1
    end
    return GTPSAPushforwardExtras(v)
  end
end

function DI.pushforward(f, ::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras{T}) where {T}
  w = f(extras.v)

  if w isa Number
    if x isa Number # Single variable scalar function derivative
      return w[1]
    else            # Multivariable scalar function derivative (gradient)
      dy = similar(w, GTPSA.numtype(eltype(w)))
      return GTPSA.gradient!(dy, w, include_params=true)
    end
  else 
    if x isa Number # Single variable vector function derivative
      
    else            # Multivariable vector function derivative (Jacobian) 
      dy = similar(w, GTPSA.numtype(eltype(w)), Size(length(w),length(x)))
    end
    dy = similar(extras.v, GTPSA.numtype(eltype(w)))
    # once again, derivatives in TPSs must be
    # 1-based linear indexing
    j = 1
    for i in eachindex(dy)
      dy[i] = w[i][j]
      j += 1
    end
    return dy
    
  end
end

function DI.value_and_pushforward(f, ::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras{T}) where {T}
  w = f(extras.v)

  if w isa Number
    return w[0], w[1]
  else  
    dy = similar(extras.v, GTPSA.numtype(eltype(w)))
    # once again, derivatives in TPSs must be
    # 1-based linear indexing
    j = 1
    for i in eachindex(dy)
      dy[i] = w[i][j]
      j += 1
    end
    return map(t->t[0], w), dy
  end
end

function DI.pushforward!(f, dy, ::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras{T}) where {T}
  w = f(extras.v)

  if w isa Number
    return w[1] # this should never be reached when this ! function is called
  else
    # once again, derivatives in TPSs must be
    # 1-based linear indexing
    j = 1
    for i in eachindex(dy)
      dy[i] = w[i][j]
      j += 1
    end
    return dy
  end
end

function DI.value_and_pushforward!(f, dy, ::AutoGTPSA, x, dx, extras::GTPSAPushforwardExtras{T}) where {T}
  w = f(extras.v)

  if w isa Number
    return w[0], w[1] # this should never be reached when this ! function is called
  else
    # once again, derivatives in TPSs must be
    # 1-based linear indexing
    j = 1
    for i in eachindex(dy)
      dy[i] = w[i][j]
      j += 1
    end
    return map(t->t[0], w), dy
  end
end

# --- derivative ---
"""
    GTPSADerivativeExtras{T} <: PushforwardExtras

This struct contains pre-allocated TPS(s) corresponding to `x` with seed
`dx` for the derivative. If taking derivative of single-variable function, 
then T is a TPS, else T is some vector of TPSs. 
"""
struct GTPSAPushforwardExtras{T} <: PushforwardExtras
  v::T
end
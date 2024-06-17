module DifferentiationInterfaceGTPSAExt

using ADTypes: AbstractADType, ForwardMode
import ADTypes: mode
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    NoDerivativeExtras,
    NoSecondDerivativeExtras,
    PushforwardExtras
using GTPSA

struct AutoGTPSA <: AbstractADType end

mode(::AutoGTPSA) = ForwardMode()

DI.check_available(::AutoGTPSA) = true

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
    return w[1]
  else  
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



end # module

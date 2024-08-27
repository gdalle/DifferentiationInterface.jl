abstract type Extras end

"""
    PushforwardExtras

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardExtras <: Extras end
struct NoPushforwardExtras <: PushforwardExtras end

"""
    PullbackExtras

Abstract type for additional information needed by [`pullback`](@ref) and its variants.
"""
abstract type PullbackExtras <: Extras end
struct NoPullbackExtras <: PullbackExtras end

"""
    DerivativeExtras

Abstract type for additional information needed by [`derivative`](@ref) and its variants.
"""
abstract type DerivativeExtras <: Extras end
struct NoDerivativeExtras <: DerivativeExtras end

"""
    GradientExtras

Abstract type for additional information needed by [`gradient`](@ref) and its variants.
"""
abstract type GradientExtras <: Extras end
struct NoGradientExtras <: GradientExtras end

"""
    JacobianExtras

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianExtras <: Extras end
struct NoJacobianExtras <: JacobianExtras end

"""
    HVPExtras

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPExtras <: Extras end
struct NoHVPExtras <: HVPExtras end

"""
    HessianExtras

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianExtras <: Extras end
struct NoHessianExtras <: HessianExtras end

"""
    SecondDerivativeExtras

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativeExtras <: Extras end
struct NoSecondDerivativeExtras <: SecondDerivativeExtras end

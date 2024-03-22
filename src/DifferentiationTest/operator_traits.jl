abstract type AbstractOperator end
abstract type AbstractFirstOrderOperator <: AbstractOperator end
abstract type AbstractSecondOrderOperator <: AbstractOperator end

# First-Order Operators
struct Pullback{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Pushforward{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Gradient{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Multiderivative{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Jacobian{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Derivative{M<:MutationBehavior} <: AbstractFirstOrderOperator end

const PullbackAllocating        = Pullback{MutationNotSupported}
const PullbackMutating          = Pullback{MutationSupported}
const PushforwardAllocating     = Pushforward{MutationNotSupported}
const PushforwardMutating       = Pushforward{MutationSupported}
const GradientAllocating        = Gradient{MutationNotSupported}
const GradientMutating          = Gradient{MutationSupported}
const MultiderivativeAllocating = Multiderivative{MutationNotSupported}
const MultiderivativeMutating   = Multiderivative{MutationSupported}
const JacobianAllocating        = Jacobian{MutationNotSupported}
const JacobianMutating          = Jacobian{MutationSupported}
const DerivativeAllocating      = Derivative{MutationNotSupported}
const DerivativeMutating        = Derivative{MutationSupported}

# Second-order operators
struct SecondDerivative{M<:MutationBehavior} <: AbstractSecondOrderOperator end
struct Hessian{M<:MutationBehavior} <: AbstractSecondOrderOperator end
struct HessianVectorProduct{M<:MutationBehavior} <: AbstractSecondOrderOperator end

const SecondDerivativeAllocating     = SecondDerivative{MutationNotSupported}
const SecondDerivativeMutating       = SecondDerivative{MutationSupported}
const HessianAllocating              = Hessian{MutationNotSupported}
const HessianMutating                = Hessian{MutationSupported}
const HessianVectorProductAllocating = HessianVectorProduct{MutationNotSupported}
const HessianVectorProductMutating   = HessianVectorProduct{MutationSupported}

## Utilities
# order
isfirstorder(::AbstractOperator)           = false
isfirstorder(::AbstractFirstOrderOperator) = true

issecondorder(::AbstractOperator)            = false
issecondorder(::AbstractSecondOrderOperator) = true

# allocations
ismutating(::Type{<:MutationBehavior}) = false
ismutating(::Type{MutationSupported})  = true

ismutating(::Pullback{M}) where {M}             = ismutating(M)
ismutating(::Pushforward{M}) where {M}          = ismutating(M)
ismutating(::Gradient{M}) where {M}             = ismutating(M)
ismutating(::Multiderivative{M}) where {M}      = ismutating(M)
ismutating(::Derivative{M}) where {M}           = ismutating(M)
ismutating(::Jacobian{M}) where {M}             = ismutating(M)
ismutating(::Hessian{M}) where {M}              = ismutating(M)
ismutating(::SecondDerivative{M}) where {M}     = ismutating(M)
ismutating(::HessianVectorProduct{M}) where {M} = ismutating(M)

isallocating(op) = !ismutating(op)

# input-output compatibility
iscompatible(op::AbstractOperator, x, y) = false
iscompatible(op::Pullback, x, y)         = true
iscompatible(op::Pushforward, x, y)      = true

iscompatible(op::Gradient, x::AbstractArray, y::Number)        = true
iscompatible(op::Multiderivative, x::Number, y::AbstractArray) = true
iscompatible(op::Derivative, x::Number, y::Number)             = true
iscompatible(op::Jacobian, x::AbstractArray, y::AbstractArray) = true
iscompatible(op::SecondDerivative, x::Number, y::Number)       = true
iscompatible(op::Hessian, x::AbstractArray, y::Number)         = true

module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian

DI.check_available(::AutoZygote) = true
DI.inplace_support(::AutoZygote) = DI.InPlaceNotSupported()

## Pullback

struct ZygotePullbackPrepSamePoint{Y,PB} <: DI.PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f, ::AutoZygote, x, ty::NTuple, contexts::Vararg{DI.ConstantOrFunctionOrBackend,C}
) where {C}
    return DI.NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f,
    ::DI.NoPullbackPrep,
    ::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, pb = pullback(f, x, map(DI.unwrap, contexts)...)
    return ZygotePullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f,
    ::DI.NoPullbackPrep,
    ::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, pb = pullback(f, x, map(DI.unwrap, contexts)...)
    tx = map(ty) do dy
        first(pb(dy))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    ::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    (; y, pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    ::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    (; pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return tx
end

## Gradient

function DI.prepare_gradient(
    f, ::AutoZygote, x, contexts::Vararg{DI.ConstantOrFunctionOrBackend,C}
) where {C}
    return DI.NoGradientPrep()
end

function DI.value_and_gradient(
    f,
    ::DI.NoGradientPrep,
    ::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    (; val, grad) = withgradient(f, x, map(DI.unwrap, contexts)...)
    return val, first(grad)
end

function DI.gradient(
    f,
    ::DI.NoGradientPrep,
    ::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return first(gradient(f, x, map(DI.unwrap, contexts)...))
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    if isnothing(new_grad)
        error("Exact zero gradient detected via Zygote `nothing` derivative. This is indicative of a coding error making the function `f` not a function of the `x` being differentiated")
    end
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

## Jacobian

function DI.prepare_jacobian(
    f, ::AutoZygote, x, contexts::Vararg{DI.ConstantOrFunctionOrBackend,C}
) where {C}
    return DI.NoJacobianPrep()
end

function DI.value_and_jacobian(
    f,
    ::DI.NoJacobianPrep,
    ::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return f(x, map(DI.unwrap, contexts)...),
    first(jacobian(f, x, map(DI.unwrap, contexts)...))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(
    f,
    ::DI.NoJacobianPrep,
    ::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return first(jacobian(f, x, map(DI.unwrap, contexts)...))
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::DI.NoJacobianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(
    f,
    jac,
    prep::DI.NoJacobianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return copyto!(jac, DI.jacobian(f, prep, backend, x, contexts...))
end

## HVP

# Beware, this uses ForwardDiff for the inner differentiation

function DI.prepare_hvp(
    f,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return DI.prepare_hvp(f, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...)
end

function DI.hvp(
    f,
    prep::DI.HVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return DI.hvp(f, prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...)
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::DI.HVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return DI.hvp!(
        f, tg, prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp(
    f,
    prep::DI.HVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return DI.gradient_and_hvp(
        f, prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp!(
    f,
    grad,
    tg::NTuple,
    prep::DI.HVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return DI.gradient_and_hvp!(
        f, grad, tg, prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

## Hessian

function DI.prepare_hessian(
    f, ::AutoZygote, x, contexts::Vararg{DI.ConstantOrFunctionOrBackend,C}
) where {C}
    return DI.NoHessianPrep()
end

function DI.hessian(
    f,
    ::DI.NoHessianPrep,
    ::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return hessian(fc, x)
end

function DI.hessian!(
    f,
    hess,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    return copyto!(hess, DI.hessian(f, prep, backend, x, contexts...))
end

function DI.value_gradient_and_hessian(
    f,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, grad = DI.value_and_gradient(f, DI.NoGradientPrep(), backend, x, contexts...)
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.ConstantOrFunctionOrBackend,C},
) where {C}
    y, _ = DI.value_and_gradient!(f, grad, DI.NoGradientPrep(), backend, x, contexts...)
    DI.hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

end

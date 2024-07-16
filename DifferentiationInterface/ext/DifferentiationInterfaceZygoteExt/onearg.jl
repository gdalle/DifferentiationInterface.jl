## Pullback

struct ZygotePullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoZygote, x, dy) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, ::AutoZygote, x, dy, ::PullbackExtras=NoPullbackExtras()
)
    y, pb = pullback(f, x)
    return ZygotePullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(f, ::AutoZygote, x, dy, ::NoPullbackExtras)
    y, pb = pullback(f, x)
    return y, only(pb(dy))
end

function DI.value_and_pullback(
    f, ::AutoZygote, x, dy, extras::ZygotePullbackExtrasSamePoint
)
    @compat (; y, pb) = extras
    return copy(y), only(pb(dy))
end

function DI.pullback(f, ::AutoZygote, x, dy, extras::ZygotePullbackExtrasSamePoint)
    @compat (; pb) = extras
    return only(pb(dy))
end

## Gradient

DI.prepare_gradient(f, ::AutoZygote, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AutoZygote, x, ::NoGradientExtras)
    @compat (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::AutoZygote, x, ::NoGradientExtras)
    return only(gradient(f, x))
end

function DI.value_and_gradient!(f, grad, backend::AutoZygote, x, extras::NoGradientExtras)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, backend::AutoZygote, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

## Jacobian

DI.prepare_jacobian(f, ::AutoZygote, x) = NoJacobianExtras()

function DI.value_and_jacobian(f, ::AutoZygote, x, ::NoJacobianExtras)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AutoZygote, x, ::NoJacobianExtras)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!(f, jac, backend::AutoZygote, x, extras::NoJacobianExtras)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(f, jac, backend::AutoZygote, x, extras::NoJacobianExtras)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

## HVP

# Beware, this uses ForwardDiff for the inner differentiation

struct ZygoteHVPExtras{G,PE} <: HVPExtras
    ∇f::G
    pushforward_extras::PE
end

function DI.prepare_hvp(f, ::AutoZygote, x, dx)
    ∇f(x) = only(gradient(f, x))
    pushforward_extras = DI.prepare_pushforward(∇f, AutoForwardDiff(), x, dx)
    return ZygoteHVPExtras(∇f, pushforward_extras)
end

function DI.hvp(f, ::AutoZygote, x, dx, extras::ZygoteHVPExtras)
    @compat (; ∇f, pushforward_extras) = extras
    return DI.pushforward(∇f, AutoForwardDiff(), x, dx, pushforward_extras)
end

function DI.hvp!(f, dg, ::AutoZygote, x, dx, extras::ZygoteHVPExtras)
    @compat (; ∇f, pushforward_extras) = extras
    return DI.pushforward!(∇f, dg, AutoForwardDiff(), x, dx, pushforward_extras)
end

struct ZygoteHVPBatchedExtras{G,PE} <: HVPExtras
    ∇f::G
    pushforward_batched_extras::PE
end

function DI.prepare_hvp_batched(f, ::AutoZygote, x, dx::Batch)
    ∇f(x) = only(gradient(f, x))
    pushforward_batched_extras = DI.prepare_pushforward_batched(
        ∇f, AutoForwardDiff(), x, dx
    )
    return ZygoteHVPBatchedExtras(∇f, pushforward_batched_extras)
end

function DI.hvp_batched(f, ::AutoZygote, x, dx::Batch, extras::ZygoteHVPBatchedExtras)
    @compat (; ∇f, pushforward_batched_extras) = extras
    return DI.pushforward_batched(∇f, AutoForwardDiff(), x, dx, pushforward_batched_extras)
end

function DI.hvp_batched!(
    f, dg::Batch, ::AutoZygote, x, dx::Batch, extras::ZygoteHVPBatchedExtras
)
    @compat (; ∇f, pushforward_batched_extras) = extras
    return DI.pushforward_batched!(
        ∇f, dg, AutoForwardDiff(), x, dx, pushforward_batched_extras
    )
end

## Hessian

DI.prepare_hessian(f, ::AutoZygote, x) = NoHessianExtras()

function DI.hessian(f, ::AutoZygote, x, ::NoHessianExtras)
    return hessian(f, x)
end

function DI.hessian!(f, hess, backend::AutoZygote, x, extras::NoHessianExtras)
    return copyto!(hess, DI.hessian(f, backend, x, extras))
end

function DI.value_gradient_and_hessian(f, backend::AutoZygote, x, extras::NoHessianExtras)
    y, grad = DI.value_and_gradient(f, backend, x, NoGradientExtras())
    hess = DI.hessian(f, backend, x, extras)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, backend::AutoZygote, x, extras::NoHessianExtras
)
    y, _ = DI.value_and_gradient!(f, grad, backend, x, NoGradientExtras())
    DI.hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end

## Second derivative

DI.prepare_second_derivative(f, ::AutoEnzyme, x, v) = NoSecondDerivativeExtras()

function DI.second_derivative(
    f, backend::AutoForwardOrNothingEnzyme, x, ::NoSecondDerivativeExtras
)
    df = DeferredDerivative(f, forward_mode(backend))
    return DI.derivative(df, AutoEnzyme(forward_mode(backend)), x)
end

function DI.second_derivative!(
    f, der2, backend::AutoForwardOrNothingEnzyme, x, ::NoSecondDerivativeExtras
)
    df = DeferredDerivative(f, forward_mode(backend))
    return DI.derivative!(df, der2, AutoEnzyme(forward_mode(backend)), x)
end

## Hessian

struct Enzyme1HessianExtras{G,JE} <: HessianExtras
    ∇f::G
    jac_extras::JE
end

function DI.prepare_hessian(f, backend::AutoEnzyme, x)
    ∇f = DeferredGradient(f, reverse_mode(backend))
    jac_extras = DI.prepare_jacobian(∇f, AutoEnzyme(forward_mode(backend)), x)
    return Enzyme1HessianExtras(∇f, jac_extras)
end

function DI.hessian(f, backend::AutoEnzyme, x, extras::Enzyme1HessianExtras)
    @compat (; ∇f, jac_extras) = extras
    return DI.jacobian(∇f, AutoEnzyme(forward_mode(backend)), x, jac_extras)
end

function DI.hessian!(f, hess, backend::AutoEnzyme, x, extras::Enzyme1HessianExtras)
    @compat (; ∇f, jac_extras) = extras
    return DI.jacobian!(∇f, hess, AutoEnzyme(forward_mode(backend)), x, jac_extras)
end

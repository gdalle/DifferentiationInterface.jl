struct ReactantGradientPrep{F,G} <: GradientPrep
    compiled_gradient::F
    compiled_value_and_gradient::G
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    xr = to_rarray(x)
    gradient_closure(xr) = DI.gradient(f, rebackend.backend, xr)
    value_and_gradient_closure(xr) = DI.value_and_gradient(f, rebackend.backend, xr)
    compiled_gradient = @compile gradient_closure(xr)
    compiled_value_and_gradient = @compile gradient_closure(xr)
    return ReactantGradientPrep(compiled_gradient, compiled_value_and_gradient)
end

function DI.gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    @compat (; compiled_gradient) = prep
    xr = to_rarray(x)
    return compiled_gradient(xr)
end

function DI.value_and_gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    @compat (; compiled_value_and_gradient) = prep
    xr = to_rarray(x)
    return compiled_value_and_gradient(xr)
end

function DI.gradient!(f, grad, prep::ReactantGradientPrep, rebackend::ReactantBackend, x)
    gradr = DI.gradient(f, prep, rebackend, x)
    return copyto!(grad, gradr)
end

function DI.value_and_gradient!(
    f, grad, prep::ReactantGradientPrep, rebackend::ReactantBackend, x
)
    y, gradr = DI.value_and_gradient(f, prep, rebackend, x)
    return y, copyto!(grad, gradr)
end

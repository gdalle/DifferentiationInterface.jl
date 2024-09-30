struct ReactantGradientPrep{F,G} <: GradientPrep
    compiled_function::F
    compiled_gradient::G
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    xr = ConcreteRArray(x)
    gradient_closure(xr) = DI.gradient(f, rebackend.backend, xr)
    compiled_function = compile(f, (xr,))
    compiled_gradient = compile(gradient_closure, (xr,))
    return ReactantGradientPrep(compiled_function, compiled_gradient)
end

function DI.gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    @compat (; compiled_gradient) = prep
    xr = ConcreteRArray(x)
    return compiled_gradient(xr)
end

function DI.value_and_gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    @compat (; compiled_function, compiled_gradient) = prep
    xr = ConcreteRArray(x)
    return compiled_function(xr), compiled_gradient(xr)
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

struct ReactantGradientExtras{F,G} <: GradientExtras
    compiled_function::F
    compiled_gradient::G
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    xr = ConcreteRArray(x)
    gradient_extras = DI.prepare_gradient(f, rebackend.backend, xr)
    gradient_closure(xr) = DI.gradient(f, rebackend.backend, xr, gradient_extras)
    compiled_function = compile(f, (xr,))
    compiled_gradient = compile(gradient_closure, (xr,))
    return ReactantGradientExtras(compiled_function, compiled_gradient)
end

function DI.gradient(f, ::ReactantBackend, x, extras::ReactantGradientExtras)
    @compat (; compiled_gradient) = extras
    xr = ConcreteRArray(x)
    return compiled_gradient(xr)
end

function DI.value_and_gradient(f, ::ReactantBackend, x, extras::ReactantGradientExtras)
    @compat (; compiled_function, compiled_gradient) = extras
    xr = ConcreteRArray(x)
    return compiled_function(xr), compiled_gradient(xr)
end

function DI.gradient!(
    f, grad, rebackend::ReactantBackend, x, extras::ReactantGradientExtras
)
    gradr = DI.gradient(f, rebackend, x, extras)
    return copyto!(grad, gradr)
end

function DI.value_and_gradient!(
    f, grad, rebackend::ReactantBackend, x, extras::ReactantGradientExtras
)
    y, gradr = DI.value_and_gradient(f, rebackend, x, extras)
    return y, copyto!(grad, gradr)
end

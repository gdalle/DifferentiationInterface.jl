struct ReactantGradientExtras{G}
    compiled_gradient_closure::G
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    xr = ConcreteRArray(x)
    gradient_extras = DI.prepare_gradient(f, rebackend.backend, xr)
    gradient_closure(xr) = DI.gradient(f, rebackend.backend, xr, gradient_extras)
    compiled_gradient_closure = compile(gradient_closure, (xr,))
    return ReactantGradientExtras(compiled_gradient_closure)
end

function DI.gradient(f, rebackend::ReactantBackend, x, extras::ReactantGradientExtras)
    @compat (; compiled_gradient_closure) = extras
    xr = ConcreteRArray(x)
    return compiled_gradient_closure(xr)
end

function DI.gradient!(
    f, grad, rebackend::ReactantBackend, x, extras::ReactantGradientExtras
)
    @compat (; compiled_gradient_closure) = extras
    xr = ConcreteRArray(x)
    gradr = compiled_gradient_closure(xr)
    return copyto!(grad, gradr)
end

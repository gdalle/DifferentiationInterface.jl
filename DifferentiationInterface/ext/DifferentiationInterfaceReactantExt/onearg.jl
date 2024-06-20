struct ReactantGradientExtras{FC,E}
    f_compiled::FC
    gradient_extras::E
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    xr = ConcreteRArray(x)
    f_compiled = compile(f, (xr,))  
    gradient_extras = DI.prepare_gradient(f_compiled, rebackend.backend, xr)
    return ReactantGradientExtras(f_compiled, gradient_extras)
end

function DI.gradient(f, rebackend::ReactantBackend, x, extras::ReactantGradientExtras)
    @compat (; f_compiled, gradient_extras) = extras
    xr = ConcreteRArray(x)
    return DI.gradient(f_compiled, rebackend.backend, xr, gradient_extras)
end

function DI.gradient!(f, grad, rebackend::ReactantBackend, x, extras::ReactantGradientExtras)
    @compat (; f_compiled, gradient_extras) = extras
    xr = ConcreteRArray(x)
    gradr = DI.gradient(f_compiled, rebackend.backend, xr, gradient_extras)
    return copyto!(grad, gradr)
end

## Pushforward

function test_jet(ba::AbstractADType, scen::Scenario{:pushforward,1,:outofplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x, seed)

    if Bool(pushforward_performance(ba))
        JET.@test_opt value_and_pushforward(f, extras, ba, x, seed)
        JET.@test_opt pushforward(f, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pushforward,1,:inplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x, seed)
    dy_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt value_and_pushforward!(f, dy_in, extras, ba, x, seed)
        JET.@test_opt pushforward!(f, dy_in, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pushforward,2,:outofplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, mysimilar(y), ba, x, seed)
    y_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt value_and_pushforward(f!, y_in, extras, ba, x, seed)
        JET.@test_opt pushforward(f!, y_in, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pushforward,2,:inplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, mysimilar(y), ba, x, seed)
    y_in, dy_in = mysimilar(y), mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt value_and_pushforward!(f!, y_in, dy_in, extras, ba, x, seed)
        JET.@test_opt pushforward!(f!, y_in, dy_in, extras, ba, x, seed)
    end
    return nothing
end

## Pullback

function test_jet(ba::AbstractADType, scen::Scenario{:pullback,1,:outofplace})
    @compat (; f, x, seed) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x, seed)

    if Bool(pullback_performance(ba))
        JET.@test_opt value_and_pullback(f, extras, ba, x, seed)
        JET.@test_opt pullback(f, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pullback,1,:inplace})
    @compat (; f, x, seed) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x, seed)
    dx_in = mysimilar(x)

    if Bool(pullback_performance(ba))
        JET.@test_opt value_and_pullback!(f, dx_in, extras, ba, x, seed)
        JET.@test_opt pullback!(f, dx_in, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pullback,2,:outofplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, mysimilar(y), ba, x, seed)
    y_in = mysimilar(y)

    if Bool(pullback_performance(ba))
        JET.@test_opt value_and_pullback(f!, y_in, extras, ba, x, seed)
        JET.@test_opt pullback(f!, y_in, extras, ba, x, seed)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:pullback,2,:inplace})
    @compat (; f, x, y, seed) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, mysimilar(y), ba, x, seed)
    y_in, dx_in = mysimilar(y), mysimilar(x)

    if Bool(pullback_performance(ba))
        JET.@test_opt value_and_pullback!(f!, y_in, dx_in, extras, ba, x, seed)
        JET.@test_opt pullback!(f!, y_in, dx_in, extras, ba, x, seed)
    end
    return nothing
end

## Derivative

function test_jet(ba::AbstractADType, scen::Scenario{:derivative,1,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)

    JET.@test_opt value_and_derivative(f, extras, ba, x)
    JET.@test_opt derivative(f, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:derivative,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = mysimilar(y)

    JET.@test_opt value_and_derivative!(f, der_in, extras, ba, x)
    JET.@test_opt derivative!(f, der_in, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:derivative,2,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, x)
    y_in = mysimilar(y)

    JET.@test_opt value_and_derivative(f!, y_in, extras, ba, x)
    JET.@test_opt derivative(f!, y_in, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:derivative,2,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, x)
    y_in, der_in = mysimilar(y), mysimilar(y)

    JET.@test_opt value_and_derivative!(f!, y_in, der_in, extras, ba, x)
    JET.@test_opt derivative!(f!, y_in, der_in, extras, ba, x)
    return nothing
end

## Gradient

function test_jet(ba::AbstractADType, scen::Scenario{:gradient,1,:outofplace})
    @compat (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)

    JET.@test_opt value_and_gradient(f, extras, ba, x)
    JET.@test_opt gradient(f, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:gradient,1,:inplace})
    @compat (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = mysimilar(x)

    JET.@test_opt value_and_gradient!(f, grad_in, extras, ba, x)
    JET.@test_opt gradient!(f, grad_in, extras, ba, x)
    return nothing
end

## Jacobian

function test_jet(ba::AbstractADType, scen::Scenario{:jacobian,1,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)

    JET.@test_opt value_and_jacobian(f, extras, ba, x)
    JET.@test_opt jacobian(f, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:jacobian,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = mysimilar(jacobian(f, ba, x))

    JET.@test_opt value_and_jacobian!(f, jac_in, extras, ba, x)
    JET.@test_opt jacobian!(f, jac_in, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:jacobian,2,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    y_in = mysimilar(y)

    JET.@test_opt value_and_jacobian(f!, y_in, extras, ba, x)
    JET.@test_opt jacobian(f!, y_in, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:jacobian,2,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    y_in, jac_in = mysimilar(y), mysimilar(jacobian(f!, mysimilar(y), ba, x))

    JET.@test_opt value_and_jacobian!(f!, y_in, jac_in, extras, ba, x)
    JET.@test_opt jacobian!(f!, y_in, jac_in, extras, ba, x)
    return nothing
end

## Second derivative

function test_jet(ba::AbstractADType, scen::Scenario{:second_derivative,1,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    JET.@test_opt second_derivative(f, extras, ba, x)
    JET.@test_opt value_derivative_and_second_derivative(f, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:second_derivative,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    der1_in = mysimilar(y)
    der2_in = mysimilar(y)

    JET.@test_opt second_derivative!(f, der2_in, extras, ba, x)
    JET.@test_opt value_derivative_and_second_derivative!(
        f, der1_in, der2_in, ba, x, extras
    )
    return nothing
end

## HVP

function test_jet(ba::AbstractADType, scen::Scenario{:hvp,1,:outofplace})
    @compat (; f, x, seed) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, seed)

    JET.@test_opt hvp(f, extras, ba, x, seed)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:hvp,1,:inplace})
    @compat (; f, x, seed) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, seed)
    p_in = mysimilar(x)

    JET.@test_opt hvp!(f, p_in, extras, ba, x, seed)
    return nothing
end

## Hessian

function test_jet(ba::AbstractADType, scen::Scenario{:hessian,1,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    JET.@test_opt hessian(f, extras, ba, x)
    JET.@test_opt value_gradient_and_hessian(f, extras, ba, x)
    return nothing
end

function test_jet(ba::AbstractADType, scen::Scenario{:hessian,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    grad_in = mysimilar(x)
    hess_in = mysimilar(hessian(f, ba, x))

    JET.@test_opt hessian!(f, hess_in, extras, ba, x)
    JET.@test_opt value_gradient_and_hessian!(f, grad_in, hess_in, extras, ba, x)
    return nothing
end

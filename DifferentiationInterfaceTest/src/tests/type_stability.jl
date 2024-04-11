## Pushforward

function test_jet(ba::AbstractADType, scen::PushforwardScenario{1,:outofplace}; ref_backend)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)

    if Bool(pushforward_performance(ba))
        @test_opt value_and_pushforward(f, ba, x, dx, extras)
        @test_opt pushforward(f, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{1,:inplace}; ref_backend)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        @test_opt value_and_pushforward!(f, dy_in, ba, x, dx, extras)
        @test_opt pushforward!(f, dy_in, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{2,:outofplace}; ref_backend)
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        @test_opt value_and_pushforward(f!, y_in, ba, x, dx, extras)
        @test_opt pushforward(f!, y_in, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{2,:inplace}; ref_backend)
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in, dy_in = mysimilar(y), mysimilar(y)

    if Bool(pushforward_performance(ba))
        @test_opt value_and_pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
        @test_opt pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
    end
    return nothing
end

## Pullback

function test_jet(ba::AbstractADType, scen::PullbackScenario{1,:outofplace}; ref_backend)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)

    _, pullbackfunc = value_and_pullback_split(f, ba, x, extras)

    if Bool(pullback_performance(ba))
        @test_opt value_and_pullback(f, ba, x, dy, extras)
        @test_opt pullback(f, ba, x, dy, extras)
        @test_opt pullbackfunc(dy)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{1,:inplace}; ref_backend)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = mysimilar(x)

    _, pullbackfunc! = value_and_pullback!_split(f, ba, x, extras)

    if Bool(pullback_performance(ba))
        @test_opt value_and_pullback!(f, dx_in, ba, x, dy, extras)
        @test_opt pullback!(f, dx_in, ba, x, dy, extras)
        @test_opt pullbackfunc!(dx_in, dy)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{2,:outofplace}; ref_backend)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = mysimilar(y)

    _, pullbackfunc = value_and_pullback_split(f!, y, ba, x, extras)

    if Bool(pullback_performance(ba))
        @test_opt value_and_pullback(f!, y_in, ba, x, dy, extras)
        @test_opt pullback(f!, y_in, ba, x, dy, extras)
        @test_opt pullbackfunc(y_in, dy)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{2,:inplace}; ref_backend)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in, dx_in = mysimilar(y), mysimilar(x)

    _, pullbackfunc! = value_and_pullback!_split(f!, y, ba, x, extras)

    if Bool(pullback_performance(ba))
        @test_opt value_and_pullback!(f!, y_in, dx_in, ba, x, dy, extras)
        @test_opt pullback!(f!, y_in, dx_in, ba, x, dy, extras)
        @test_opt pullbackfunc!(y_in, dx_in, dy)
    end
    return nothing
end

## Derivative

function test_jet(ba::AbstractADType, scen::DerivativeScenario{1,:outofplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)

    @test_opt value_and_derivative(f, ba, x, extras)
    @test_opt derivative(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{1,:inplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = mysimilar(y)

    @test_opt value_and_derivative!(f, der_in, ba, x, extras)
    @test_opt derivative!(f, der_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{2,:outofplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = mysimilar(y)

    @test_opt value_and_derivative(f!, y_in, ba, x, extras)
    @test_opt derivative(f!, y_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{2,:inplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in, der_in = mysimilar(y), mysimilar(y)

    @test_opt value_and_derivative!(f!, (y_in, der_in), ba, x, extras)
    @test_opt derivative!(f!, (y_in, der_in), ba, x, extras)
    return nothing
end

## Gradient

function test_jet(ba::AbstractADType, scen::GradientScenario{1,:outofplace}; ref_backend)
    (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)

    @test_opt value_and_gradient(f, ba, x, extras)
    @test_opt gradient(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::GradientScenario{1,:inplace}; ref_backend)
    (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = mysimilar(x)

    @test_opt value_and_gradient!(f, grad_in, ba, x, extras)
    @test_opt gradient!(f, grad_in, ba, x, extras)
    return nothing
end

## Jacobian

function test_jet(ba::AbstractADType, scen::JacobianScenario{1,:outofplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)

    @test_opt value_and_jacobian(f, ba, x, extras)
    @test_opt jacobian(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{1,:inplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = Matrix{eltype(y)}(undef, length(y), length(x))

    @test_opt value_and_jacobian!(f, jac_in, ba, x, extras)
    @test_opt jacobian!(f, jac_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{2,:outofplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = mysimilar(y)

    @test_opt value_and_jacobian(f!, y_in, ba, x, extras)
    @test_opt jacobian(f!, y_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{2,:inplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in, jac_in = mysimilar(y), Matrix{eltype(y)}(undef, length(y), length(x))

    @test_opt value_and_jacobian!(f!, y_in, jac_in, ba, x, extras)
    @test_opt jacobian!(f!, y_in, jac_in, ba, x, extras)
    return nothing
end

## Second derivative

function test_jet(
    ba::AbstractADType, scen::SecondDerivativeScenario{1,:outofplace}; ref_backend
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    @test_opt second_derivative(f, ba, x, extras)
    return nothing
end

function test_jet(
    ba::AbstractADType, scen::SecondDerivativeScenario{1,:inplace}; ref_backend
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    der2_in = mysimilar(y)

    @test_opt second_derivative!(f, der2_in, ba, x, extras)
    return nothing
end

## HVP

function test_jet(ba::AbstractADType, scen::HVPScenario{1,:outofplace}; ref_backend)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)

    @test_opt hvp(f, ba, x, dx, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::HVPScenario{1,:inplace}; ref_backend)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)
    p_in = mysimilar(x)

    @test_opt hvp!(f, p_in, ba, x, dx, extras)
    return nothing
end

## Hessian

function test_jet(ba::AbstractADType, scen::HessianScenario{1,:outofplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    @test_opt hessian(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::HessianScenario{1,:inplace}; ref_backend)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_in = Matrix{typeof(y)}(undef, length(x), length(x))

    @test_opt hessian!(f, hess_in, ba, x, extras)
    return nothing
end

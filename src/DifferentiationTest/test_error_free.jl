
function test_error_free(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false})
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = mysimilar(dy)

    @test (value_and_pushforward!!(f, dy_in, ba, x, dx, extras); true)
    @test (value_and_pushforward(f, ba, x, dx, extras); true)
    return nothing
end

function test_error_free(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true})
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = mysimilar(y)
    dy_in = mysimilar(dy)
    @test (value_and_pushforward!!(f!, y_in, dy_in, ba, x, dx, extras); true)
    return nothing
end

## Pullback

function test_error_free(ba::AbstractADType, ::typeof(pullback), scen::Scenario{false})
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = mysimilar(dx)

    @test (value_and_pullback!!(f, dx_in, ba, x, dy, extras); true)
    @test (value_and_pullback(f, ba, x, dy, extras); true)
    return nothing
end

function test_error_free(ba::AbstractADType, ::typeof(pullback), scen::Scenario{true})
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = mysimilar(y)
    dx_in = mysimilar(dx)

    @test (value_and_pullback!!(f!, y_in, dx_in, ba, x, dy, extras); true)
    return nothing
end

## Derivative

function test_error_free(ba::AbstractADType, ::typeof(derivative), scen::Scenario{false})
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = mysimilar(dy)

    @test (value_and_derivative!!(f, der_in, ba, x, extras); true)
    @test (value_and_derivative(f, ba, x, extras); true)
    return nothing
end

function test_error_free(ba::AbstractADType, ::typeof(derivative), scen::Scenario{true})
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = mysimilar(y)
    der_in = mysimilar(dy)

    @test (value_and_derivative!!(f!, y_in, der_in, ba, x, extras); true)
    return nothing
end

## Gradient

function test_error_free(ba::AbstractADType, ::typeof(gradient), scen::Scenario{false})
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = mysimilar(dx)

    @test (value_and_gradient!!(f, grad_in, ba, x, extras); true)
    @test (value_and_gradient(f, ba, x, extras); true)
    return nothing
end

## Jacobian

function test_error_free(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = similar(x, length(y), length(x))

    @test (value_and_jacobian!!(f, jac_in, ba, x, extras); true)
    @test (value_and_jacobian(f, ba, x, extras); true)
    return nothing
end

function test_error_free(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true})
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = mysimilar(y)
    jac_in = similar(x, length(y), length(x))

    @test (value_and_jacobian!!(f!, y_in, jac_in, ba, x, extras); true)
    return nothing
end

## Second derivative

function test_error_free(
    ba::AbstractADType, ::typeof(second_derivative), scen::Scenario{false};
)
    (; f, x) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    @test (second_derivative(f, ba, x, extras); true)
    return nothing
end

## HVP

function test_error_free(ba::AbstractADType, ::typeof(hvp), scen::Scenario{false})
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)

    @test (hvp(f, ba, x, dx, extras); true)
    return nothing
end

## Hessian

function test_error_free(ba::AbstractADType, ::typeof(hessian), scen::Scenario{false})
    (; f, x) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    @test (hessian(f, ba, x, extras); true)
    return nothing
end

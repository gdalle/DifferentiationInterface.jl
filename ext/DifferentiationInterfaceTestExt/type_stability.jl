# This file requires JET

## Pushforward 

function test_type_pushforward_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, dx, dy) = deepcopy(scenario)
    dy_in = zero(dy)
    @test_opt value_and_pushforward!(dy_in, ba, f, x, dx, maybe_extras...)
    @test_opt pushforward!(dy_in, ba, f, x, dx, maybe_extras...)
    @test_opt value_and_pushforward(ba, f, x, dx, maybe_extras...)
    @test_opt pushforward(ba, f, x, dx, maybe_extras...)
end

function test_type_pushforward_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    y_in = zero(y)
    dy_in = zero(dy)
    @test_opt value_and_pushforward!(y_in, dy_in, ba, f!, x, dx, maybe_extras...)
end

## Pullback

function test_type_pullback_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, dx, dy) = deepcopy(scenario)
    dx_in = zero(dx)
    @test_opt value_and_pullback!(dx_in, ba, f, x, dy, maybe_extras...)
    @test_opt pullback!(dx_in, ba, f, x, dy, maybe_extras...)
    @test_opt value_and_pullback(ba, f, x, dy, maybe_extras...)
    @test_opt pullback(ba, f, x, dy, maybe_extras...)
end

function test_type_pullback_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    y_in = zero(y)
    dx_in = zero(dx)
    @test_opt value_and_pullback!(y_in, dx_in, ba, f!, x, dy, maybe_extras...)
end

## Derivative

function test_type_derivative_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x) = deepcopy(scenario)
    @test_opt value_and_derivative(ba, f, x, maybe_extras...)
    @test_opt derivative(ba, f, x, maybe_extras...)
end

## Multiderivative

function test_type_multiderivative_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, dy) = deepcopy(scenario)
    multider_in = zero(dy)
    @test_opt value_and_multiderivative!(multider_in, ba, f, x, maybe_extras...)
    @test_opt multiderivative!(multider_in, ba, f, x, maybe_extras...)
    @test_opt value_and_multiderivative(ba, f, x, maybe_extras...)
    @test_opt multiderivative(ba, f, x, maybe_extras...)
end

function test_type_multiderivative_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dy) = deepcopy(scenario)
    f! = f
    y_in = zero(y)
    multider_in = zero(dy)
    @test_opt value_and_multiderivative!(y_in, multider_in, ba, f!, x, maybe_extras...)
end

## Gradient

function test_type_gradient_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, dx) = deepcopy(scenario)
    grad_in = zero(dx)
    @test_opt value_and_gradient!(grad_in, ba, f, x, maybe_extras...)
    @test_opt gradient!(grad_in, ba, f, x, maybe_extras...)
    @test_opt value_and_gradient(ba, f, x, maybe_extras...)
    @test_opt gradient(ba, f, x, maybe_extras...)
end

## Hessian

function test_type_hessian_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, dx) = deepcopy(scenario)
    grad_in = zero(dx)
    hvp_in = zero(dx)
    hess_in = zeros(eltype(x), length(x), length(x))
    @test_opt value_and_gradient_and_hessian!(grad_in, hess_in, ba, f, x, maybe_extras...)
    @test_opt hessian!(hess_in, ba, f, x, maybe_extras...)
    @test_opt gradient_and_hessian_vector_product!(
        grad_in, hvp_in, ba, f, x, dx, maybe_extras...
    )
    @test_opt value_and_gradient_and_hessian(ba, f, x, maybe_extras...)
    @test_opt hessian(ba, f, x, maybe_extras...)
    @test_opt gradient_and_hessian_vector_product(ba, f, x, dx, maybe_extras...)
end

## Jacobian

function test_type_jacobian_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    jac_in = zeros(eltype(y), length(y), length(x))
    @test_opt value_and_jacobian!(jac_in, ba, f, x, maybe_extras...)
    @test_opt jacobian!(jac_in, ba, f, x, maybe_extras...)
    @test_opt value_and_jacobian(ba, f, x, maybe_extras...)
    @test_opt jacobian(ba, f, x, maybe_extras...)
end

function test_type_jacobian_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))
    @test_opt value_and_jacobian!(y_in, jac_in, ba, f!, x, maybe_extras...)
end

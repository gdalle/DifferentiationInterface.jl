# This file requires ForwardDiff

## Utils

function true_pushforward(f, x::Number, y::Number, dx; mutating)
    return ForwardDiff.derivative(f, x) * dx
end

function true_pushforward(f, x::Number, y::AbstractArray, dx; mutating)
    if mutating
        return ForwardDiff.derivative(f, deepcopy(y), x) .* dx
    else
        return ForwardDiff.derivative(f, x) .* dx
    end
end

function true_pushforward(f, x::AbstractArray, y::Number, dx; mutating)
    return dot(ForwardDiff.gradient(f, x), dx)
end

function true_pushforward(f, x::AbstractArray, y::AbstractArray, dx; mutating)
    if mutating
        return reshape(ForwardDiff.jacobian(f, deepcopy(y), x) * vec(dx), size(y))
    else
        return reshape(ForwardDiff.jacobian(f, x) * vec(dx), size(y))
    end
end

function true_pullback(f, x::Number, y::Number, dy; mutating)
    return ForwardDiff.derivative(f, x) * dy
end

function true_pullback(f, x::Number, y::AbstractArray, dy; mutating)
    if mutating
        return dot(ForwardDiff.derivative(f, deepcopy(y), x), dy)
    else
        return dot(ForwardDiff.derivative(f, x), dy)
    end
end

function true_pullback(f, x::AbstractArray, y::Number, dy; mutating)
    return ForwardDiff.gradient(f, x) .* dy
end

function true_pullback(f, x::AbstractArray, y::AbstractArray, dy; mutating)
    if mutating
        return reshape(
            transpose(ForwardDiff.jacobian(f, deepcopy(y), x)) * vec(dy), size(x)
        )
    else
        return reshape(transpose(ForwardDiff.jacobian(f, x)) * vec(dy), size(x))
    end
end

## Pushforward

function test_correctness_pushforward_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dx) = deepcopy(scenario)
    dy_true = true_pushforward(f, x, y, dx; mutating=false)

    y_out1, dy_out1 = value_and_pushforward(ba, f, x, dx, maybe_extras...)
    dy_in2 = zero(dy_out1)
    y_out2, dy_out2 = value_and_pushforward!(dy_in2, ba, f, x, dx, maybe_extras...)

    dy_out3 = pushforward(ba, f, x, dx, maybe_extras...)
    dy_in4 = zero(dy_out3)
    dy_out4 = pushforward!(dy_in4, ba, f, x, dx, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Tangent value" begin
        @test dy_out1 ≈ dy_true rtol = 1e-3
        @test dy_out2 ≈ dy_true rtol = 1e-3
        @test dy_out3 ≈ dy_true rtol = 1e-3
        @test dy_out4 ≈ dy_true rtol = 1e-3
        if ismutable(dy_true)
            @testset "Mutation" begin
                @test dy_in2 ≈ dy_true rtol = 1e-3
                @test dy_in4 ≈ dy_true rtol = 1e-3
            end
        end
    end
end

function test_correctness_pushforward_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dx) = deepcopy(scenario)
    f! = f
    dy_true = true_pushforward(f!, x, y, dx; mutating=true)

    y_in = zero(y)
    dy_in = zero(dy_true)
    y_out, dy_out = value_and_pushforward!(y_in, dy_in, ba, f!, x, dx, maybe_extras...)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            @test y_in ≈ y
        end
    end
    @testset "Tangent value" begin
        @test dy_out ≈ dy_true rtol = 1e-3
        @testset "Mutation" begin
            @test dy_in ≈ dy_true rtol = 1e-3
        end
    end
end

## Pullback

function test_correctness_pullback_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dy) = deepcopy(scenario)
    dx_true = true_pullback(f, x, y, dy; mutating=false)

    y_out1, dx_out1 = value_and_pullback(ba, f, x, dy, maybe_extras...)
    dx_in2 = zero(dx_out1)
    y_out2, dx_out2 = value_and_pullback!(dx_in2, ba, f, x, dy, maybe_extras...)

    dx_out3 = pullback(ba, f, x, dy, maybe_extras...)
    dx_in4 = zero(dx_out3)
    dx_out4 = pullback!(dx_in4, ba, f, x, dy, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Cotangent value" begin
        @test dx_out1 ≈ dx_true rtol = 1e-3
        @test dx_out2 ≈ dx_true rtol = 1e-3
        @test dx_out3 ≈ dx_true rtol = 1e-3
        @test dx_out4 ≈ dx_true rtol = 1e-3
        if ismutable(dx_true)
            @testset "Mutation" begin
                @test dx_in2 ≈ dx_true rtol = 1e-3
                @test dx_in4 ≈ dx_true rtol = 1e-3
            end
        end
    end
end

function test_correctness_pullback_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dy) = deepcopy(scenario)
    f! = f
    dx_true = true_pullback(f, x, y, dy; mutating=true)

    y_in = zero(y)
    dx_in = zero(dx_true)
    y_out, dx_out = value_and_pullback!(y_in, dx_in, ba, f!, x, dy, maybe_extras...)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            @test y_in ≈ y
        end
    end
    @testset "Cotangent value" begin
        @test dx_out ≈ dx_true rtol = 1e-3
        if ismutable(dx_true)
            @testset "Mutation" begin
                @test dx_in ≈ dx_true rtol = 1e-3
            end
        end
    end
end

## Derivative

function test_correctness_derivative_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    der_true = ForwardDiff.derivative(f, x)

    y_out1, der_out1 = value_and_derivative(ba, f, x, maybe_extras...)

    der_out2 = derivative(ba, f, x, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
    end
    @testset "Derivative value" begin
        @test der_out1 ≈ der_true rtol = 1e-3
        @test der_out2 ≈ der_true rtol = 1e-3
    end
end

## Multiderivative

function test_correctness_multiderivative_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    multider_true = ForwardDiff.derivative(f, x)

    y_out1, multider_out1 = value_and_multiderivative(ba, f, x, maybe_extras...)
    multider_in2 = zero(multider_out1)
    y_out2, multider_out2 = value_and_multiderivative!(
        multider_in2, ba, f, x, maybe_extras...
    )

    multider_out3 = multiderivative(ba, f, x, maybe_extras...)
    multider_in4 = zero(multider_out3)
    multider_out4 = multiderivative!(multider_in4, ba, f, x, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Multiderivative value" begin
        @test multider_out1 ≈ multider_true rtol = 1e-3
        @test multider_out2 ≈ multider_true rtol = 1e-3
        @test multider_out3 ≈ multider_true rtol = 1e-3
        @test multider_out4 ≈ multider_true rtol = 1e-3
        @testset "Mutation" begin
            @test multider_in2 ≈ multider_true rtol = 1e-3
            @test multider_in4 ≈ multider_true rtol = 1e-3
        end
    end
end

function test_correctness_multiderivative_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    multider_true = ForwardDiff.derivative(f!, y, x)

    y_in = zero(y)
    multider_in = zero(multider_true)
    y_out, multider_out = value_and_multiderivative!(
        y_in, multider_in, ba, f!, x, maybe_extras...
    )

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            @test y_in ≈ y
        end
    end
    @testset "Multiderivative value" begin
        @test multider_out ≈ multider_true rtol = 1e-3
        @testset "Mutation" begin
            @test multider_in ≈ multider_true rtol = 1e-3
        end
    end
end

## Gradient

function test_correctness_gradient_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    grad_true = ForwardDiff.gradient(f, x)

    y_out1, grad_out1 = value_and_gradient(ba, f, x, maybe_extras...)
    grad_in2 = zero(grad_out1)
    y_out2, grad_out2 = value_and_gradient!(grad_in2, ba, f, x, maybe_extras...)

    grad_out3 = gradient(ba, f, x, maybe_extras...)
    grad_in4 = zero(grad_out3)
    grad_out4 = gradient!(grad_in4, ba, f, x, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Gradient value" begin
        @test grad_out1 ≈ grad_true rtol = 1e-3
        @test grad_out2 ≈ grad_true rtol = 1e-3
        @test grad_out3 ≈ grad_true rtol = 1e-3
        @test grad_out4 ≈ grad_true rtol = 1e-3
        @testset "Mutation" begin
            @test grad_in2 ≈ grad_true rtol = 1e-3
            @test grad_in4 ≈ grad_true rtol = 1e-3
        end
    end
end

## Hessian

function test_correctness_hessian_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y, dx) = deepcopy(scenario)
    grad_true = ForwardDiff.gradient(f, x)
    hess_true = ForwardDiff.hessian(f, x)
    hvp_true = reshape((hess_true * vec(dx)), size(x))

    y_out1, grad_out1, hess_out1 = value_and_gradient_and_hessian(ba, f, x, maybe_extras...)
    grad_in2, hess_in2 = zero(grad_out1), zero(hess_out1)
    y_out2, grad_out2, hess_out2 = value_and_gradient_and_hessian!(
        grad_in2, hess_in2, ba, f, x, maybe_extras...
    )

    hess_out3 = hessian(ba, f, x, maybe_extras...)
    hess_in4 = zero(hess_out3)
    hess_out4 = hessian!(hess_in4, ba, f, x, maybe_extras...)

    grad_out5, hvp_out5 = gradient_and_hessian_vector_product(ba, f, x, dx, maybe_extras...)
    grad_in6, hvp_in6 = zero(grad_out5), zero(hvp_out5)
    grad_out6, hvp_out6 = gradient_and_hessian_vector_product!(
        grad_in6, hvp_in6, ba, f, x, dx, maybe_extras...
    )

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Gradient value" begin
        @test grad_out1 ≈ grad_true rtol = 1e-3
        @test grad_out2 ≈ grad_true rtol = 1e-3
        @test grad_out5 ≈ grad_true rtol = 1e-3
        @test grad_out6 ≈ grad_true rtol = 1e-3
        @testset "Mutation" begin
            @test grad_in2 ≈ grad_true rtol = 1e-3
            @test grad_in6 ≈ grad_true rtol = 1e-3
        end
    end
    @testset "Hessian value" begin
        @test hess_out1 ≈ hess_true rtol = 1e-3
        @test hess_out2 ≈ hess_true rtol = 1e-3
        @test hess_out3 ≈ hess_true rtol = 1e-3
        @test hess_out4 ≈ hess_true rtol = 1e-3
        @testset "Mutation" begin
            @test hess_in2 ≈ hess_true rtol = 1e-3
            @test hess_in4 ≈ hess_true rtol = 1e-3
        end
    end
    @testset "Hessian-vector product value" begin
        @test hvp_out5 ≈ hvp_true rtol = 1e-3
        @test hvp_out6 ≈ hvp_true rtol = 1e-3
        @testset "Mutation" begin
            @test hvp_in6 ≈ hvp_true rtol = 1e-3
        end
    end
end

## Jacobian

function test_correctness_jacobian_allocating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    jac_true = ForwardDiff.jacobian(f, x)

    y_out1, jac_out1 = value_and_jacobian(ba, f, x, maybe_extras...)
    jac_in2 = zero(jac_out1)
    y_out2, jac_out2 = value_and_jacobian!(jac_in2, ba, f, x, maybe_extras...)

    jac_out3 = jacobian(ba, f, x, maybe_extras...)
    jac_in4 = zero(jac_out3)
    jac_out4 = jacobian!(jac_in4, ba, f, x, maybe_extras...)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Jacobian value" begin
        @test jac_out1 ≈ jac_true rtol = 1e-3
        @test jac_out2 ≈ jac_true rtol = 1e-3
        @test jac_out3 ≈ jac_true rtol = 1e-3
        @test jac_out4 ≈ jac_true rtol = 1e-3
        @testset "Mutation" begin
            @test jac_in2 ≈ jac_true rtol = 1e-3
            @test jac_in4 ≈ jac_true rtol = 1e-3
        end
    end
end

function test_correctness_jacobian_mutating(
    ba::AbstractADType, scenario::Scenario, maybe_extras...
)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    jac_true = ForwardDiff.jacobian(f!, y, x)

    y_in = zero(y)
    jac_in = similar(y, length(y), length(x))
    y_out, jac_out = value_and_jacobian!(y_in, jac_in, ba, f!, x, maybe_extras...)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            @test y_in ≈ y
        end
    end
    @testset "Jacobian value" begin
        @test jac_out ≈ jac_true rtol = 1e-3
        @testset "Mutation" begin
            @test jac_in ≈ jac_true rtol = 1e-3
        end
    end
end

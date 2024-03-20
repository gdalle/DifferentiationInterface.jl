
## Selector

function DT.test_correctness(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Correctness" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_correctness_pushforward_allocating(backend, s)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_correctness_pushforward_mutating(backend, s)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_correctness_pullback_allocating(backend, s)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_correctness_pullback_mutating(backend, s)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_correctness_derivative_allocating(backend, s)
                    end

                elseif op == :multiderivative_allocating
                    @testset "$s" for s in allocating(scalar_array(scenarios))
                        test_correctness_multiderivative_allocating(backend, s)
                    end
                elseif op == :multiderivative_mutating
                    @testset "$s" for s in mutating(scalar_array(scenarios))
                        test_correctness_multiderivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_correctness_gradient_allocating(backend, s)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        test_correctness_jacobian_allocating(backend, s)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        test_correctness_jacobian_mutating(backend, s)
                    end

                elseif op == :second_derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_correctness_second_derivative_allocating(backend, s)
                    end

                elseif op == :hessian_vector_product_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_correctness_hessian_vector_product_allocating(backend, s)
                    end
                elseif op == :hessian_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_correctness_hessian_allocating(backend, s)
                    end

                else
                    throw(ArgumentError("Invalid operator to test: `:$op`"))
                end
            end
        end
    end
end

## Pushforward

function test_correctness_pushforward_allocating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, y, dx) = deepcopy(scen)
    dy_true = true_pushforward(f, x, y, dx; mutating=false)

    y_out1, dy_out1 = DI.value_and_pushforward(ba, f, x, dx)
    dy_in2 = zero(dy_out1)
    y_out2, dy_out2 = DI.value_and_pushforward!(dy_in2, ba, f, x, dx)

    dy_out3 = DI.pushforward(ba, f, x, dx)
    dy_in4 = zero(dy_out3)
    dy_out4 = DI.pushforward!(dy_in4, ba, f, x, dx)

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

function test_correctness_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dx) = deepcopy(scen)
    f! = f
    dy_true = true_pushforward(f!, x, y, dx; mutating=true)

    y_in = zero(y)
    dy_in = zero(dy_true)
    y_out, dy_out = DI.value_and_pushforward!(y_in, dy_in, ba, f!, x, dx)

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

function test_correctness_pullback_allocating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, y, dy) = deepcopy(scen)
    dx_true = true_pullback(f, x, y, dy; mutating=false)

    y_out1, dx_out1 = DI.value_and_pullback(ba, f, x, dy)
    dx_in2 = zero(dx_out1)
    y_out2, dx_out2 = DI.value_and_pullback!(dx_in2, ba, f, x, dy)

    dx_out3 = DI.pullback(ba, f, x, dy)
    dx_in4 = zero(dx_out3)
    dx_out4 = DI.pullback!(dx_in4, ba, f, x, dy)

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

function test_correctness_pullback_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    dx_true = true_pullback(f, x, y, dy; mutating=true)

    y_in = zero(y)
    dx_in = zero(dx_true)
    y_out, dx_out = DI.value_and_pullback!(y_in, dx_in, ba, f!, x, dy)

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

function test_correctness_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    der_true = ForwardDiff.derivative(f, x)

    y_out1, der_out1 = DI.value_and_derivative(ba, f, x)

    der_out2 = DI.derivative(ba, f, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
    end
    @testset "Derivative value" begin
        @test der_out1 ≈ der_true rtol = 1e-3
        @test der_out2 ≈ der_true rtol = 1e-3
    end
end

## Multiderivative

function test_correctness_multiderivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    multider_true = ForwardDiff.derivative(f, x)

    y_out1, multider_out1 = DI.value_and_multiderivative(ba, f, x)
    multider_in2 = zero(multider_out1)
    y_out2, multider_out2 = DI.value_and_multiderivative!(multider_in2, ba, f, x)

    multider_out3 = DI.multiderivative(ba, f, x)
    multider_in4 = zero(multider_out3)
    multider_out4 = DI.multiderivative!(multider_in4, ba, f, x)

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

function test_correctness_multiderivative_mutating(ba::AbstractADType, scen::Scenario)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    multider_true = ForwardDiff.derivative(f!, y, x)

    y_in = zero(y)
    multider_in = zero(multider_true)
    y_out, multider_out = DI.value_and_multiderivative!(y_in, multider_in, ba, f!, x)

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

function test_correctness_gradient_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    grad_true = ForwardDiff.gradient(f, x)

    y_out1, grad_out1 = DI.value_and_gradient(ba, f, x)
    grad_in2 = zero(grad_out1)
    y_out2, grad_out2 = DI.value_and_gradient!(grad_in2, ba, f, x)

    grad_out3 = DI.gradient(ba, f, x)
    grad_in4 = zero(grad_out3)
    grad_out4 = DI.gradient!(grad_in4, ba, f, x)

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

## Jacobian

function test_correctness_jacobian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    jac_true = ForwardDiff.jacobian(f, x)

    y_out1, jac_out1 = DI.value_and_jacobian(ba, f, x)
    jac_in2 = zero(jac_out1)
    y_out2, jac_out2 = DI.value_and_jacobian!(jac_in2, ba, f, x)

    jac_out3 = DI.jacobian(ba, f, x)
    jac_in4 = zero(jac_out3)
    jac_out4 = DI.jacobian!(jac_in4, ba, f, x)

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

function test_correctness_jacobian_mutating(ba::AbstractADType, scen::Scenario)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    jac_true = ForwardDiff.jacobian(f!, y, x)

    y_in = zero(y)
    jac_in = similar(y, length(y), length(x))
    y_out, jac_out = DI.value_and_jacobian!(y_in, jac_in, ba, f!, x)

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

## Second derivative

function test_correctness_second_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    der_true = ForwardDiff.derivative(f, x)
    derder_true = ForwardDiff.derivative(x) do z
        ForwardDiff.derivative(f, z)
    end

    y_out1, der_out1, derder_out1 = DI.value_derivative_and_second_derivative(ba, f, x)

    derder_out2 = DI.second_derivative(ba, f, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
    end
    @testset "Derivative value" begin
        @test der_out1 ≈ der_true rtol = 1e-3
    end
    @testset "Second derivative value" begin
        @test derder_out1 ≈ derder_true rtol = 1e-3
        @test derder_out2 ≈ derder_true rtol = 1e-3
    end
end

## Hessian-vector product

function test_correctness_hessian_vector_product_allocating(
    ba::AbstractADType, scen::Scenario
)
    (; f, x, dx) = deepcopy(scen)
    grad_true = ForwardDiff.gradient(f, x)
    hess_true = ForwardDiff.hessian(f, x)
    hvp_true = reshape((hess_true * vec(dx)), size(x))

    hvp_out5 = DI.hessian_vector_product(ba, f, x, dx)
    hvp_in6 = zero(hvp_out5)
    hvp_out6 = DI.hessian_vector_product!(hvp_in6, ba, f, x, dx)

    grad_out7, hvp_out7 = DI.gradient_and_hessian_vector_product(ba, f, x, dx)
    grad_in8, hvp_in8 = zero(grad_out7), zero(hvp_out7)
    grad_out8, hvp_out8 = DI.gradient_and_hessian_vector_product!(
        grad_in8, hvp_in8, ba, f, x, dx
    )

    @testset "Gradient value" begin
        @test grad_out7 ≈ grad_true rtol = 1e-3
        @test grad_out8 ≈ grad_true rtol = 1e-3
        @testset "Mutation" begin
            @test grad_in8 ≈ grad_true rtol = 1e-3
        end
    end

    @testset "Hessian-vector product value" begin
        @test hvp_out5 ≈ hvp_true rtol = 1e-3
        @test hvp_out6 ≈ hvp_true rtol = 1e-3
        @test hvp_out7 ≈ hvp_true rtol = 1e-3
        @test hvp_out8 ≈ hvp_true rtol = 1e-3
        @testset "Mutation" begin
            @test hvp_in6 ≈ hvp_true rtol = 1e-3
            @test hvp_in8 ≈ hvp_true rtol = 1e-3
        end
    end
end

## Hessian

function test_correctness_hessian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    grad_true = ForwardDiff.gradient(f, x)
    hess_true = ForwardDiff.hessian(f, x)

    y_out1, grad_out1, hess_out1 = DI.value_gradient_and_hessian(ba, f, x)
    grad_in2, hess_in2 = zero(grad_out1), zero(hess_out1)
    y_out2, grad_out2, hess_out2 = DI.value_gradient_and_hessian!(
        grad_in2, hess_in2, ba, f, x
    )

    hess_out3 = DI.hessian(ba, f, x)
    hess_in4 = zero(hess_out3)
    hess_out4 = DI.hessian!(hess_in4, ba, f, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Gradient value" begin
        @test grad_out1 ≈ grad_true rtol = 1e-3
        @test grad_out2 ≈ grad_true rtol = 1e-3
        @testset "Mutation" begin
            @test grad_in2 ≈ grad_true rtol = 1e-3
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
end

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

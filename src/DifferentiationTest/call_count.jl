struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter{F})(x) where {F}
    cc.count[] += 1
    return cc.f(x)
end

function (cc::CallCounter{F})(y, x) where {F}
    cc.count[] += 1
    return cc.f(y, x)
end

function test_call_count(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Correctness" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_call_count_pushforward_allocating(backend, s)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_call_count_pushforward_mutating(backend, s)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_call_count_pullback_allocating(backend, s)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_call_count_pullback_mutating(backend, s)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_call_count_derivative_allocating(backend, s)
                    end

                elseif op == :multiderivative_allocating
                    @testset "$s" for s in allocating(scalar_array(scenarios))
                        test_call_count_multiderivative_allocating(backend, s)
                    end
                elseif op == :multiderivative_mutating
                    @testset "$s" for s in mutating(scalar_array(scenarios))
                        test_call_count_multiderivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_call_count_gradient_allocating(backend, s)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        test_call_count_jacobian_allocating(backend, s)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        test_call_count_jacobian_mutating(backend, s)
                    end

                elseif op == :second_derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_call_count_second_derivative_allocating(backend, s)
                    end

                elseif op == :hessian_vector_product_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_call_count_hessian_vector_product_allocating(backend, s)
                    end
                elseif op == :hessian_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_call_count_hessian_allocating(backend, s)
                    end

                else
                    throw(ArgumentError("Invalid operator to test: `:$op`"))
                end
            end
        end
    end
end

## Pushforward

function test_call_count_pushforward_allocating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, dx) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pushforward(ba, cc, x, dx)
    if isa(mode(ba), ForwardMode)
        @test cc.count[] <= 1
    end
end

function test_call_count_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    dy_in = zero(dy)
    value_and_pushforward!(y_in, dy_in, ba, cc!, x, dx)
    if isa(mode(ba), ForwardMode)
        @test cc!.count[] <= 1
    end
end

## Pullback

function test_call_count_pullback_allocating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pullback(ba, cc, x, dy)
    if isa(mode(ba), ReverseMode)
        @test cc.count[] <= 1
    end
end

function test_call_count_pullback_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    dx_in = zero(dx)
    value_and_pullback!(y_in, dx_in, ba, cc!, x, dy)
    if isa(mode(ba), ReverseMode)
        @test cc!.count[] <= 1
    end
end

## Derivative

function test_call_count_derivative_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_derivative(ba, cc, x)
    @test cc.count[] <= 1
end

## Multiderivative

function test_call_count_multiderivative_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number,<:AbstractArray}
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_multiderivative(ba, cc1, x)
    multiderivative(ba, cc2, x)
    if isa(mode(ba), ForwardMode)
        @test cc1.count[] <= 1
        @test cc2.count[] <= 1
    elseif isa(mode(ba), ReverseMode)
        @test cc1.count[] <= length(y)
        @test cc2.count[] <= length(y)
    end
end

function test_call_count_multiderivative_mutating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number,<:AbstractArray}
)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    multider_in = zero(dy)
    value_and_multiderivative!(y_in, multider_in, ba, cc!, x)
    if isa(mode(ba), ForwardMode)
        @test cc!.count[] <= 1
    elseif isa(mode(ba), ReverseMode)
        @test cc!.count[] <= length(y)
    end
end

## Gradient

function test_call_count_gradient_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:Number}
)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_gradient(ba, cc1, x)
    gradient(ba, cc2, x)
    if isa(mode(ba), ForwardMode)
        @test cc1.count[] <= length(x)
        @test cc2.count[] <= length(x)
    elseif isa(mode(ba), ReverseMode)
        @test cc1.count[] <= 1
        @test cc2.count[] <= 1
    end
end

## Jacobian

function test_call_count_jacobian_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray}
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_jacobian(ba, cc1, x)
    jacobian(ba, cc2, x)
    if isa(mode(ba), ForwardMode)
        @test cc1.count[] <= 2 + length(x)  # at least one too many
        @test cc2.count[] <= 2 + length(x)  # at least one too many
    elseif isa(mode(ba), ReverseMode)
        @test cc1.count[] <= 2 + length(y)  # at least one too many
        @test cc2.count[] <= 2 + length(y)  # at least one too many
    end
end

function test_call_count_jacobian_mutating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray}
)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    jac_in = similar(y, length(y), length(x))
    value_and_jacobian!(y_in, jac_in, ba, cc!, x)
    if isa(mode(ba), ForwardMode)
        @test cc!.count[] <= 1 + length(x)
    elseif isa(mode(ba), ReverseMode)
        @test cc!.count[] <= 1 + length(y)
    end
end

## Second derivative

function test_call_count_second_derivative_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_derivative_and_second_derivative(ba, cc, x)
    @test cc.count[] <= 2
end

## Hessian-vector product

function test_call_count_hessian_vector_product_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:Number}
)
    (; f, x, dx) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    hessian_vector_product(ba, cc1, x, dx)
    gradient_and_hessian_vector_product(ba, cc2, x, dx)
    @test cc1.count[] <= 2
    @test cc2.count[] <= 2
end

## Hessian

function test_call_count_hessian_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_gradient_and_hessian(ba, cc1, x)
    hessian(ba, cc2, x)
    @test cc1.count[] <= 2 + length(x)
    @test cc2.count[] <= 2 + length(x)
end

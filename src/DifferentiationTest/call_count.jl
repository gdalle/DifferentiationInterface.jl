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
    operators::Vector{<:AbstractOperator},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Call count" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                @testset "$s" for s in filter_compatible(op, scenarios)
                    test_call_count(op, backend, s)
                end
            end
        end
    end
end

## Pushforward

function test_call_count(::PushforwardAllocating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pushforward(ba, cc, x, dx)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    end
end

function test_call_count(::PushforwardMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    dy_in = zero(dy)
    value_and_pushforward!(y_in, dy_in, ba, cc!, x, dx)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    end
end

## Pullback

function test_call_count(::PullbackAllocating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pullback(ba, cc, x, dy)
    if mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

function test_call_count(::PullbackMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    dx_in = zero(dx)
    value_and_pullback!(y_in, dx_in, ba, cc!, x, dy)
    if mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1
    end
end

## Derivative

function test_call_count(
    ::DerivativeAllocating, ba::AbstractADType, scen::Scenario{<:Any,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_derivative(ba, cc, x)
    @test cc.count[] <= 1
end

## Multiderivative

function test_call_count(
    ::MultiderivativeAllocating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:Number,<:AbstractArray},
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_multiderivative(ba, cc1, x)
    multiderivative(ba, cc2, x)
    if mode(ba) == AbstractForwardMode
        @test cc1.count[] <= 1
        @test cc2.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc1.count[] <= length(y)
        @test cc2.count[] <= length(y)
    end
end

function test_call_count(
    ::MultiderivativeMutating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:Number,<:AbstractArray},
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    multider_in = zero(dy)
    value_and_multiderivative!(y_in, multider_in, ba, cc!, x)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= length(y)
    end
end

## Gradient

function test_call_count(
    ::GradientAllocating, ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_gradient(ba, cc1, x)
    gradient(ba, cc2, x)
    if mode(ba) == AbstractForwardMode
        @test cc1.count[] <= length(x)
        @test cc2.count[] <= length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc1.count[] <= 1
        @test cc2.count[] <= 1
    end
end

## Jacobian

function test_call_count(
    ::JacobianAllocating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray},
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_jacobian(ba, cc1, x)
    jacobian(ba, cc2, x)
    if mode(ba) == AbstractForwardMode
        @test cc1.count[] <= 2 + length(x)  # at least one too many
        @test cc2.count[] <= 2 + length(x)  # at least one too many
    elseif mode(ba) == AbstractReverseMode
        @test cc1.count[] <= 2 + length(y)  # at least one too many
        @test cc2.count[] <= 2 + length(y)  # at least one too many
    end
end

function test_call_count(
    ::JacobianMutating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray},
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = zero(y)
    jac_in = similar(y, length(y), length(x))
    value_and_jacobian!(y_in, jac_in, ba, cc!, x)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1 + length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1 + length(y)
    end
end

## Second derivative

function test_call_count(
    ::SecondDerivativeAllocating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:Number,<:Number},
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_derivative_and_second_derivative(ba, cc, x)
    @test cc.count[] <= 2
end

## Hessian-vector product

function test_call_count(
    ::HessianVectorProductAllocating,
    ba::AbstractADType,
    scen::Scenario{<:Any,<:AbstractArray,<:Number},
)
    Bool(supports_hvp(ba)) || return nothing
    (; f, x, dx) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    hessian_vector_product(ba, cc1, x, dx)
    # gradient_and_hessian_vector_product(ba, cc2, x, dx)
    @test cc1.count[] <= 2
    # @test cc2.count[] <= 2
end

## Hessian

function test_call_count(
    ::HessianAllocating, ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc1 = CallCounter(f)
    cc2 = CallCounter(f)
    value_gradient_and_hessian(ba, cc1, x)
    hessian(ba, cc2, x)
    @test cc1.count[] <= 2 + length(x)
    @test cc2.count[] <= 2 + length(x)
end

function test_call_count(op::AbstractOperator, ba::AbstractADType, scen::Scenario)
    throw(ArgumentError("Invalid operator to test: $op"))
end

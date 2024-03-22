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
                    @testset "$s" for s in allocating(scalar_in(scenarios))
                        test_call_count_derivative_allocating(backend, s)
                    end
                elseif op == :derivative_mutating
                    @testset "$s" for s in mutating(scalar_in(scenarios))
                        test_call_count_derivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(scalar_out(scenarios))
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

                else
                    throw(ArgumentError("Invalid operator to test: `:$op`"))
                end
            end
        end
    end
end

## Pushforward

function test_call_count_pushforward_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pushforward(cc, ba, x, dx)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    end
end

function test_call_count_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = myzero(y)
    dy_in = myzero(dy)
    value_and_pushforward!(cc!, y_in, dy_in, ba, x, dx)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    end
end

## Pullback

function test_call_count_pullback_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_pullback(cc, ba, x, dy)
    if mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

function test_call_count_pullback_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = myzero(y)
    dx_in = myzero(dx)
    value_and_pullback!(cc!, y_in, dx_in, ba, x, dy)
    if mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1
    end
end

## Derivative

function test_call_count_derivative_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number,<:Any}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_derivative(cc, ba, x)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= length(y)
    end
end

function test_call_count_derivative_mutating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Number,<:AbstractArray}
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = myzero(y)
    der_in = myzero(dy)
    value_and_derivative!(cc!, y_in, der_in, ba, x)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= length(y)
    end
end

## Gradient

function test_call_count_gradient_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:Any,<:Number}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    value_and_gradient(cc, ba, x)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

## Jacobian

function test_call_count_jacobian_allocating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray}
)
    (; f, x, y) = deepcopy(scen)
    cc = CallCounter(f)
    cc2 = CallCounter(f)
    value_and_jacobian(cc, ba, x)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 2 + length(x)  # at least one too many
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= 2 + length(y)  # at least one too many
    end
end

function test_call_count_jacobian_mutating(
    ba::AbstractADType, scen::Scenario{<:Any,<:AbstractArray,<:AbstractArray}
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    cc! = CallCounter(f)
    y_in = myzero(y)
    jac_in = similar(y, length(y), length(x))
    value_and_jacobian!(cc!, y_in, jac_in, ba, x)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1 + length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1 + length(y)
    end
end

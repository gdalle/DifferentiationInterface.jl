struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter)(x)
    cc.count[] += 1
    return cc.f(x)
end

function (cc::CallCounter)(y, x)
    cc.count[] += 1
    return cc.f(y, x)
end

## Pushforward

function test_call_count(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false})
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_pushforward(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_pushforward(cc, ba, x, dx, extras)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    end
end

function test_call_count(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true})
    (; f, x, y, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = mysimilar(y)
    dy_in = mysimilar(dy)
    value_and_pushforward!!(cc!, y_in, dy_in, ba, x, dx, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    end
end

## Pullback

function test_call_count(ba::AbstractADType, ::typeof(pullback), scen::Scenario{false})
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_pullback(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_pullback(cc, ba, x, dy, extras)
    if mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

function test_call_count(ba::AbstractADType, ::typeof(pullback), scen::Scenario{true})
    (; f, x, y, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = mysimilar(y)
    dx_in = mysimilar(dx)
    value_and_pullback!!(cc!, y_in, dx_in, ba, x, dy, extras)
    if mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1
    end
end

## Derivative

function test_call_count(ba::AbstractADType, scen::DerivativeScenario{false})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_derivative(cc, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= length(y)
    end
end

function test_call_count(ba::AbstractADType, scen::DerivativeScenario{true})
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_derivative(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = mysimilar(y)
    der_in = mysimilar(dy)
    value_and_derivative!!(cc!, y_in, der_in, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= length(y)
    end
end

## Gradient

function test_call_count(ba::AbstractADType, scen::GradientScenario{false})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_gradient(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_gradient(cc, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

## Jacobian

function test_call_count(ba::AbstractADType, scen::JacobianScenario{false})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_jacobian(cc, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 2 + length(x)  # at least one too many
    elseif mode(ba) == AbstractReverseMode
        @test cc.count[] <= 2 + length(y)  # at least one too many
    end
end

function test_call_count(ba::AbstractADType, scen::JacobianScenario{true})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = mysimilar(y)
    jac_in = Matrix{eltype(y)}(undef, length(y), length(x))
    value_and_jacobian!!(cc!, y_in, jac_in, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1 + length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1 + length(y)
    end
end

## Second derivative

function test_call_count(ba::AbstractADType, scen::SecondDerivativeScenario{false})
    (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(CallCounter(f), ba, x)
    cc = CallCounter(f)
    second_derivative(cc, ba, x, extras)
    # what to test?
    return nothing
end

## Hessian-vector product

function test_call_count(ba::AbstractADType, scen::HVPScenario{false})
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hvp(CallCounter(f), ba, x)
    cc = CallCounter(f)
    hvp(cc, ba, x, dx, extras)
    # what to test?
    return nothing
end

## Hessian

function test_call_count(ba::AbstractADType, scen::HessianScenario{false})
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hessian(CallCounter(f), ba, x)
    cc = CallCounter(f)
    hessian(cc, ba, x, extras)
    # what to test?
    return nothing
end

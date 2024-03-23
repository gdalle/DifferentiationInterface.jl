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

## Pushforward

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_pushforward), scen::Scenario{false}
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_pushforward(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_pushforward(cc, ba, x, dx, extras)
    if mode(ba) == AbstractForwardMode
        @test cc.count[] <= 1
    end
end

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_pushforward), scen::Scenario{true}
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = myzero(y)
    dy_in = myzero(dy)
    value_and_pushforward!(cc!, y_in, dy_in, ba, x, dx, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    end
end

## Pullback

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_pullback), scen::Scenario{false}
)
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_pullback(CallCounter(f), ba, x)
    cc = CallCounter(f)
    value_and_pullback(cc, ba, x, dy, extras)
    if mode(ba) == AbstractReverseMode
        @test cc.count[] <= 1
    end
end

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_pullback), scen::Scenario{true}
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = myzero(y)
    dx_in = myzero(dx)
    value_and_pullback!(cc!, y_in, dx_in, ba, x, dy, extras)
    if mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1
    end
end

## Derivative

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_derivative), scen::Scenario{false}
)
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

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_derivative), scen::Scenario{true}
)
    (; f, x, y, dy) = deepcopy(scen)
    extras = prepare_derivative(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = myzero(y)
    der_in = myzero(dy)
    value_and_derivative!(cc!, y_in, der_in, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= length(y)
    end
end

## Gradient

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_gradient), scen::Scenario{false}
)
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

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_jacobian), scen::Scenario{false}
)
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

function test_call_count(
    ba::AbstractADType, ::typeof(value_and_jacobian), scen::Scenario{true}
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(CallCounter(f), ba, y, x)
    cc! = CallCounter(f)
    y_in = myzero(y)
    jac_in = similar(y, length(y), length(x))
    value_and_jacobian!(cc!, y_in, jac_in, ba, x, extras)
    if mode(ba) == AbstractForwardMode
        @test cc!.count[] <= 1 + length(x)
    elseif mode(ba) == AbstractReverseMode
        @test cc!.count[] <= 1 + length(y)
    end
end

function test_call_count(ba::AbstractADType, op::Function, scen::Scenario)
    throw(ArgumentError("Invalid operator to test: $op"))
end

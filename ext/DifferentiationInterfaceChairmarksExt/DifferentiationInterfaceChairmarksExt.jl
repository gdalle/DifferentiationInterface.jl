module DifferentiationInterfaceChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface: myzero
using DifferentiationInterface.DifferentiationTest: Scenario, BenchmarkData, record!
import DifferentiationInterface.DifferentiationTest as DT
using Test: @testset, @test

## Pushforward

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_pushforward),
    scen::Scenario{false};
    allocations::Bool,
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    bench1 = @be myzero(dy) value_and_pushforward!(f, _, ba, x, dx, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_pushforward),
    scen::Scenario{true};
    allocations::Bool,
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    bench1 = @be (myzero(y), myzero(dy)) value_and_pushforward!(
        f!, _[1], _[2], ba, x, dx, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

## Pullback

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_pullback),
    scen::Scenario{false};
    allocations::Bool,
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    bench1 = @be myzero(dx) value_and_pullback!(f, _, ba, x, dy, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_pullback),
    scen::Scenario{true};
    allocations::Bool,
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    bench1 = @be (myzero(y), myzero(dx)) value_and_pullback!(
        f!, _[1], _[2], ba, x, dy, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

## Derivative

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_derivative),
    scen::Scenario{false};
    allocations::Bool,
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    bench1 = @be myzero(dy) value_and_derivative!(f, _, ba, x, extras)
    # never test allocations
    record!(data, ba, op, scen, bench1)
    return nothing
end

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_derivative),
    scen::Scenario{true};
    allocations::Bool,
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    bench1 = @be (myzero(y), myzero(dy)) value_and_derivative!(
        f!, _[1], _[2], ba, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

## Gradient

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_gradient),
    scen::Scenario{false};
    allocations::Bool,
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    bench1 = @be myzero(dx) value_and_gradient!(f, _, ba, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

## Jacobian

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_jacobian),
    scen::Scenario{false};
    allocations::Bool,
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_template = zeros(eltype(y), length(y), length(x))
    bench1 = @be myzero(jac_template) value_and_jacobian!(f, _, ba, x, extras)
    # never test allocations
    record!(data, ba, op, scen, bench1)
    return nothing
end

function DT.run_benchmark!(
    data::BenchmarkData,
    ba::AbstractADType,
    op::typeof(value_and_jacobian),
    scen::Scenario{true};
    allocations::Bool,
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    jac_template = zeros(eltype(y), length(y), length(x))
    bench1 = @be (myzero(y), myzero(jac_template)) value_and_jacobian!(
        f!, _[1], _[2], ba, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, op, scen, bench1)
    return nothing
end

end # module

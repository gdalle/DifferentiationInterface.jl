module DifferentiationInterfaceChairmarksExt

using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface:
    inner,
    mode,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    supports_hvp
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using DifferentiationInterface.DifferentiationTest:
    AbstractOperator,
    PushforwardAllocating,
    PushforwardMutating,
    PullbackAllocating,
    PullbackMutating,
    MultiderivativeAllocating,
    MultiderivativeMutating,
    GradientAllocating,
    JacobianAllocating,
    JacobianMutating,
    DerivativeAllocating,
    SecondDerivativeAllocating,
    HessianAllocating,
    HessianVectorProductAllocating,
    compatible_scenarios
using Test

function DT.run_benchmark(
    backends::Vector{<:AbstractADType},
    operators::Vector{<:AbstractOperator},
    scenarios::Vector{<:Scenario};
    allocations=false,
)
    data = BenchmarkData()
    @testset verbose = true "Allocations" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                @testset "$s" for s in compatible_scenarios(op, scenarios)
                    benchmark!(op, data, backend, s; allocations)
                end
            end
        end
    end
    return data
end

## Pushforward

function benchmark!(
    ::PushforwardAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)

    extras = prepare_pushforward(ba, f, x)
    bench1 = @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
    bench2 = @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    record!(data, ba, scen, :pushforward!, bench2)
    return nothing
end

function benchmark!(
    ::PushforwardMutating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_pushforward!(
        _[1], _[2], ba, f!, x, dx, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    return nothing
end

## Pullback

function benchmark!(
    ::PullbackAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(ba, f, x)
    bench1 = @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
    bench2 = @be zero(dx) pullback!(_, ba, f, x, dy, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    record!(data, ba, scen, :pullback!, bench2)
    return nothing
end

function benchmark!(
    ::PullbackMutating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    return nothing
end

## Derivative

function benchmark!(
    ::DerivativeAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x) = deepcopy(scen)
    extras = prepare_derivative(ba, f, x)
    bench1 = @be value_and_derivative(ba, f, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_derivative, bench1)
    return nothing
end

## Multiderivative

function benchmark!(
    ::MultiderivativeAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_multiderivative(ba, f, x)
    bench1 = @be zero(dy) value_and_multiderivative!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_multiderivative!, bench1)
    return nothing
end

function benchmark!(
    ::MultiderivativeMutating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_multiderivative(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_multiderivative!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_multiderivative!, bench1)
    return nothing
end

## Gradient

function benchmark!(
    ::GradientAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(ba, f, x)
    bench1 = @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
    bench2 = @be zero(dx) gradient!(_, ba, f, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_gradient!, bench1)
    record!(data, ba, scen, :gradient!, bench2)
    return nothing
end

## Jacobian

function benchmark!(
    ::JacobianAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x, y) = deepcopy(scen)
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f, x)
    bench1 = @be zero(jac_template) value_and_jacobian!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

function benchmark!(
    ::JacobianMutating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    bench1 = @be (zero(y), zero(jac_template)) value_and_jacobian!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

## Second derivative

function benchmark!(
    ::SecondDerivativeAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x) = deepcopy(scen)
    extras = prepare_second_derivative(ba, f, x)
    bench1 = @be value_derivative_and_second_derivative(ba, f, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_derivative_and_second_derivative, bench1)
    return nothing
end

## Hessian-vector product

function benchmark!(
    ::HessianVectorProductAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    Bool(supports_hvp(ba)) || return nothing
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hessian_vector_product(ba, f, x)
    # bench1 = @be (zero(dx), zero(dx)) gradient_and_hessian_vector_product!(
    #     _[1], _[2], ba, f, x, dx, extras
    # )
    bench2 = @be zero(dx) hessian_vector_product!(_, ba, f, x, dx, extras)
    if allocations  # TODO: distinguish
        # @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    # record!(data, ba, scen, :gradient_and_hessian_vector_product!, bench1)
    record!(data, ba, scen, :hessian_vector_product!, bench2)
    return nothing
end

## Hessian

function benchmark!(
    ::HessianAllocating,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hessian(ba, f, x)
    hess_template = zeros(eltype(y), length(x), length(x))
    bench1 = @be (zero(dx), zero(hess_template)) value_gradient_and_hessian!(
        _[1], _[2], ba, f, x, extras
    )
    bench2 = @be (zero(hess_template)) hessian!(_, ba, f, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_gradient_and_hessian!, bench1)
    record!(data, ba, scen, :hessian!, bench2)
    return nothing
end

function benchmark!(
    op::AbstractOperator,
    data::BenchmarkData,
    ba::AbstractADType,
    scen::Scenario;
    allocations::Bool,
)
    throw(ArgumentError("Invalid operator to test: $op"))
end

end # module

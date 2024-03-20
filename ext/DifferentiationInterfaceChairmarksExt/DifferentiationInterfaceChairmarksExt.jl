module DifferentiationInterfaceChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface:
    ForwardMode,
    ReverseMode,
    MutationSupported,
    MutationNotSupported,
    mode,
    mutation_behavior,
    outer
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using Test

function soft_test_zero(v)
    if iszero(v)
        @test v == 0
    else
        @test_broken v == 0
    end
end

function DT.run_benchmark(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
    allocations=false,
)
    data = BenchmarkData()
    @testset verbose = true "Allocations" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        benchmark_pushforward_allocating!(data, backend, s; allocations)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        benchmark_pushforward_mutating!(data, backend, s; allocations)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        benchmark_pullback_allocating!(data, backend, s; allocations)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        benchmark_pullback_mutating!(data, backend, s; allocations)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        benchmark_derivative_allocating!(data, backend, s; allocations)
                    end

                elseif op == :multiderivative_allocating
                    @testset "$s" for s in allocating(scalar_array(scenarios))
                        benchmark_multiderivative_allocating!(data, backend, s; allocations)
                    end
                elseif op == :multiderivative_mutating
                    @testset "$s" for s in mutating(scalar_array(scenarios))
                        benchmark_multiderivative_mutating!(data, backend, s; allocations)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        benchmark_gradient_allocating!(data, backend, s; allocations)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        benchmark_jacobian_allocating!(data, backend, s; allocations)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        benchmark_jacobian_mutating!(data, backend, s; allocations)
                    end

                elseif op == :second_derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        benchmark_second_derivative_allocating!(
                            data, backend, s; allocations
                        )
                    end

                elseif op == :hessian_vector_product_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        benchmark_hessian_vector_product_allocating!(
                            data, backend, s; allocations
                        )
                    end
                elseif op == :hessian_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        benchmark_hessian_allocating!(data, backend, s; allocations)
                    end

                else
                    throw(ArgumentError("Invalid operator to benchmark: `:$op`"))
                end
            end
        end
    end
    return data
end

## Pushforward

function benchmark_pushforward_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scen)

    extras = prepare_pushforward(ba, f, x)
    bench1 = @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
    bench2 = @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
    if allocations && dy isa Number
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    record!(data, ba, scen, :pushforward!, bench2)
    return nothing
end

function benchmark_pushforward_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_pushforward!(
        _[1], _[2], ba, f!, x, dx, extras
    )
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    return nothing
end

## Pullback

function benchmark_pullback_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(ba, f, x)
    bench1 = @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
    bench2 = @be zero(dx) pullback!(_, ba, f, x, dy, extras)
    if allocations && dy isa Number
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    record!(data, ba, scen, :pullback!, bench2)
    return nothing
end

function benchmark_pullback_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    return nothing
end

## Derivative

function benchmark_derivative_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x) = deepcopy(scen)
    extras = prepare_derivative(ba, f, x)
    bench1 = @be value_and_derivative(ba, f, x, extras)
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_and_derivative, bench1)
    return nothing
end

## Multiderivative

function benchmark_multiderivative_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_multiderivative(ba, f, x)
    bench1 = @be zero(dy) value_and_multiderivative!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_multiderivative!, bench1)
    return nothing
end

function benchmark_multiderivative_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_multiderivative(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_multiderivative!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_and_multiderivative!, bench1)
    return nothing
end

## Gradient

function benchmark_gradient_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(ba, f, x)
    bench1 = @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
    bench2 = @be zero(dx) gradient!(_, ba, f, x, extras)
    if allocations
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    record!(data, ba, scen, :value_and_gradient!, bench1)
    record!(data, ba, scen, :gradient!, bench2)
    return nothing
end

## Jacobian

function benchmark_jacobian_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f, x)
    bench1 = @be zero(jac_template) value_and_jacobian!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

function benchmark_jacobian_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    bench1 = @be (zero(y), zero(jac_template)) value_and_jacobian!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

## Second derivative

function benchmark_second_derivative_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x) = deepcopy(scen)
    extras = prepare_second_derivative(ba, f, x)
    bench1 = @be value_derivative_and_second_derivative(ba, f, x, extras)
    if allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    record!(data, ba, scen, :value_derivative_and_second_derivative, bench1)
    return nothing
end

## Hessian-vector product

function benchmark_hessian_vector_product_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hessian_vector_product(ba, f, x)
    bench1 = @be (zero(dx), zero(dx)) gradient_and_hessian_vector_product!(
        _[1], _[2], ba, f, x, dx, extras
    )
    bench2 = @be zero(dx) hessian_vector_product!(_, ba, f, x, dx, extras)
    if allocations  # TODO: distinguish
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    record!(data, ba, scen, :gradient_and_hessian_vector_product!, bench1)
    record!(data, ba, scen, :hessian_vector_product!, bench2)
    return nothing
end

## Hessian

function benchmark_hessian_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_hessian(ba, f, x)
    hess_template = zeros(eltype(y), length(x), length(x))
    bench1 = @be (zero(dx), zero(hess_template)) value_gradient_and_hessian!(
        _[1], _[2], ba, f, x, extras
    )
    bench2 = @be (zero(hess_template)) hessian!(_, ba, f, x, extras)
    if allocations
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    record!(data, ba, scen, :value_gradient_and_hessian!, bench1)
    record!(data, ba, scen, :hessian!, bench2)
    return nothing
end

end

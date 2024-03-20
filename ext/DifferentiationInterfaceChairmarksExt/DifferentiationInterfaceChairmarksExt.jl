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

## Dict of run_benchmarks

struct BenchmarkDict{D}
    d::D
end

Base.show(io::IO, bd::BenchmarkDict) = print(io, bd.d)

BenchmarkDict() = BenchmarkDict(Dict())
BenchmarkDict(args...) = BenchmarkDict(Dict(args...))

Base.isempty(bd::BenchmarkDict) = isempty(bd.d)
Base.keys(bd::BenchmarkDict) = keys(bd.d)
Base.values(bd::BenchmarkDict, k) = values(bd.d)
Base.getindex(bd::BenchmarkDict, k) = get!(bd.d, k, BenchmarkDict())
Base.getindex(bd::BenchmarkDict, k...) = bd[k[1:(end - 1)]...][k[end]]
Base.setindex!(bd::BenchmarkDict, v, k) = setindex!(bd.d, v, k)

function Base.merge!(bd1::BenchmarkDict, d2::Dict)
    bd = BenchmarkDict(merge!(bd1.d, d2))
    return bd
end

function soft_test_zero(v)
    if iszero(v)
        @test v == 0
    else
        @test_broken v == 0
    end
end

## Selector

function scen_id(s::Scenario)
    return (Symbol(s.f), s.mutating, typeof(s.x), typeof(s.y), size(s.x), size(s.y))
end

function DT.run_benchmark(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
    test_allocations=false,
)
    all_results = BenchmarkDict()
    @testset verbose = true "Allocations" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                results = all_results[backend_string(backend)][op]
                if op == :pushforward_allocating
                    @testset "$(scen_string(s))" for s in allocating(scenarios)
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_pushforward_allocating(
                                backend, s; test_allocations
                            ),
                        )
                    end
                elseif op == :pushforward_mutating
                    @testset "$(scen_string(s))" for s in mutating(scenarios)
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_pushforward_mutating(
                                backend, s; test_allocations
                            ),
                        )
                    end

                elseif op == :pullback_allocating
                    @testset "$(scen_string(s))" for s in allocating(scenarios)
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_pullback_allocating(backend, s; test_allocations),
                        )
                    end
                elseif op == :pullback_mutating
                    @testset "$(scen_string(s))" for s in mutating(scenarios)
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_pullback_mutating(backend, s; test_allocations),
                        )
                    end

                elseif op == :derivative_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(scalar_scalar(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_derivative_allocating(
                                backend, s; test_allocations
                            ),
                        )
                    end

                elseif op == :multiderivative_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(scalar_array(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_multiderivative_allocating(
                                backend, s; test_allocations
                            ),
                        )
                    end
                elseif op == :multiderivative_mutating
                    @testset "$(scen_string(s))" for s in mutating(scalar_array(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_multiderivative_mutating(
                                backend, s; test_allocations
                            ),
                        )
                    end

                elseif op == :gradient_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(array_scalar(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_gradient_allocating(backend, s; test_allocations),
                        )
                    end

                elseif op == :jacobian_allocating
                    @testset "$(scen_string(s))" for s in allocating(array_array(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_jacobian_allocating(backend, s; test_allocations),
                        )
                    end
                elseif op == :jacobian_mutating
                    @testset "$(scen_string(s))" for s in mutating(array_array(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_jacobian_mutating(backend, s; test_allocations),
                        )
                    end

                elseif op == :second_derivative_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(scalar_scalar(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_second_derivative_allocating(
                                backend, s; test_allocations
                            ),
                        )
                    end

                elseif op == :hessian_vector_product_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(array_scalar(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_hessian_vector_product_allocating(
                                backend, s; test_allocations
                            ),
                        )
                    end
                elseif op == :hessian_allocating
                    @testset "$(scen_string(s))" for s in
                                                     allocating(array_scalar(scenarios))
                        merge!(
                            results[scen_id(s)...],
                            run_benchmark_hessian_allocating(backend, s; test_allocations),
                        )
                    end

                else
                    throw(ArgumentError("Invalid operator to run_benchmark: `:$op`"))
                end
            end
        end
    end
    return all_results
end

## Pushforward

function run_benchmark_pushforward_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    isa(mode(ba), ReverseMode) && return Dict()
    (; f, x, dx, dy) = deepcopy(scenario)

    extras = prepare_pushforward(ba, f, x)
    bench1 = @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
    bench2 = @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
    if test_allocations && dy isa Number
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    return Dict(:value_and_pushforward! => bench1, :pushforward! => bench2)
end

function run_benchmark_pushforward_mutating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    isa(mode(ba), ReverseMode) && return Dict()
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_pushforward!(
        _[1], _[2], ba, f!, x, dx, extras
    )
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_and_pushforward! => bench1)
end

## Pullback

function run_benchmark_pullback_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    isa(mode(ba), ForwardMode) && return Dict()
    (; f, x, dx, dy) = deepcopy(scenario)
    extras = prepare_pullback(ba, f, x)
    bench1 = @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
    bench2 = @be zero(dx) pullback!(_, ba, f, x, dy, extras)
    if test_allocations && dy isa Number
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    return Dict(:value_and_pullback! => bench1, :pullback! => bench2)
end

function run_benchmark_pullback_mutating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    isa(mode(ba), ForwardMode) && return Dict()
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_and_pullback! => bench1)
end

## Derivative

function run_benchmark_derivative_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x) = deepcopy(scenario)
    extras = prepare_derivative(ba, f, x)
    bench1 = @be value_and_derivative(ba, f, x, extras)
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_and_derivative => bench1)
end

## Multiderivative

function run_benchmark_multiderivative_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, dy) = deepcopy(scenario)
    extras = prepare_multiderivative(ba, f, x)
    bench1 = @be zero(dy) value_and_multiderivative!(_, ba, f, x, extras)
    # never test allocations
    return Dict(:value_and_multiderivative! => bench1)
end

function run_benchmark_multiderivative_mutating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, y, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_multiderivative(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_multiderivative!(
        _[1], _[2], ba, f!, x, extras
    )
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_and_multiderivative! => bench1)
end

## Gradient

function run_benchmark_gradient_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_gradient(ba, f, x)
    bench1 = @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
    bench2 = @be zero(dx) gradient!(_, ba, f, x, extras)
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    return Dict(:value_and_gradient! => bench1, :gradient! => bench2)
end

## Jacobian

function run_benchmark_jacobian_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, y) = deepcopy(scenario)
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f, x)
    bench1 = @be zero(jac_template) value_and_jacobian!(_, ba, f, x, extras)
    # never test allocations
    return Dict(:value_and_jacobian! => bench1)
end

function run_benchmark_jacobian_mutating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    bench1 = @be (zero(y), zero(jac_template)) value_and_jacobian!(
        _[1], _[2], ba, f!, x, extras
    )
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_and_jacobian! => bench1)
end

## Second derivative

function run_benchmark_second_derivative_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x) = deepcopy(scenario)
    extras = prepare_second_derivative(ba, f, x)
    bench1 = @be value_derivative_and_second_derivative(ba, f, x, extras)
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
    end
    return Dict(:value_derivative_and_second_derivative => bench1)
end

## Hessian-vector product

function run_benchmark_hessian_vector_product_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_hessian_vector_product(ba, f, x)
    bench1 = @be zero(dx) hessian_vector_product!(_, ba, f, x, dx, extras)
    bench2 = @be (zero(dx), zero(dx)) gradient_and_hessian_vector_product!(
        _[1], _[2], ba, f, x, dx, extras
    )
    if test_allocations  # TODO: distinguish
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    return Dict(
        :hessian_vector_product! => bench1, :gradient_and_hessian_vector_product! => bench2
    )
end

## Hessian

function run_benchmark_hessian_allocating(
    ba::AbstractADType, scenario::Scenario; test_allocations::Bool
)
    (; f, x, y, dx) = deepcopy(scenario)
    extras = prepare_hessian(ba, f, x)
    hess_template = zeros(eltype(y), length(x), length(x))
    bench1 = @be (zero(dx), zero(hess_template)) value_gradient_and_hessian!(
        _[1], _[2], ba, f, x, extras
    )
    bench2 = @be (zero(hess_template)) hessian!(_, ba, f, x, extras)
    if test_allocations
        soft_test_zero(minimum(bench1).allocs)
        soft_test_zero(minimum(bench2).allocs)
    end
    return Dict(:value_gradient_and_hessian! => bench1, :hessian! => bench2)
end

end

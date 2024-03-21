using ADTypes
using ADTypes:
    AbstractADType, AbstractFiniteDifferencesMode, AbstractForwardMode, AbstractReverseMode
using BenchmarkTools
using DifferentiationInterface
using DifferentiationInterface: MutationSupported, mode, mutation
using DifferentiationInterface.DifferentiationTest

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

pretty_extras(::Nothing) = "unprepared"
pretty_extras(something) = "prepared"

## Mutation

handles_mutation(b::AbstractADType) = supports_mutation(b) == MutationSupported()

## Benchmark suite

function add_pushforward_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? randn() : randn(n)
    dy = m == 1 ? 0.0 : zeros(m)

    if mode(backend) == AbstractReverseMode
        return nothing
    end

    for extras in unique([nothing, prepare_pushforward(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_pushforward"] = @benchmarkable begin
            value_and_pushforward($backend, $f, $x, $dx, $extras)
        end
        subgroup["value_and_pushforward!"] = @benchmarkable begin
            value_and_pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end

        subgroup["pushforward"] = @benchmarkable begin
            pushforward($backend, $f, $x, $dx, $extras)
        end
        subgroup["pushforward!"] = @benchmarkable begin
            pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end
    end

    if dy isa AbstractArray && handles_mutation(backend)
        y = similar(dy)
        for extras in unique([nothing, prepare_pushforward(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][backend_string(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_pushforward!"] = @benchmarkable begin
                value_and_pushforward!($y, $dy, $backend, $f, $x, $dx, $extras)
            end
        end
    end

    return nothing
end

function add_pullback_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? 0.0 : zeros(n)
    dy = m == 1 ? randn() : randn(m)

    if mode(backend) in (AbstractForwardMode, AbstractFiniteDifferencesMode)
        return nothing
    end

    for extras in unique([nothing, prepare_pullback(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_pullback"] = @benchmarkable begin
            value_and_pullback($backend, $f, $x, $dy, $extras)
        end
        subgroup["value_and_pullback!"] = @benchmarkable begin
            value_and_pullback!($dx, $backend, $f, $x, $dy, $extras)
        end

        subgroup["pullback"] = @benchmarkable begin
            pullback($backend, $f, $x, $dy, $extras)
        end
        subgroup["pullback!"] = @benchmarkable begin
            pullback!($dx, $backend, $f, $x, $dy, $extras)
        end
    end

    if dy isa AbstractArray && handles_mutation(backend)
        y = similar(dy)
        for extras in unique([nothing, prepare_pullback(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][backend_string(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_pullback!"] = @benchmarkable begin
                value_and_pullback!($y, $dx, $backend, $f, $x, $dy, $extras)
            end
        end
    end

    return nothing
end

function add_derivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == m == 1

    x = randn()

    for extras in unique([nothing, prepare_derivative(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_derivative"] = @benchmarkable begin
            value_and_derivative($backend, $f, $x, $extras)
        end

        subgroup["derivative"] = @benchmarkable begin
            derivative($backend, $f, $x, $extras)
        end
    end

    return nothing
end

function add_multiderivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == 1

    x = randn()
    multider = zeros(m)

    for extras in unique([nothing, prepare_multiderivative(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_multiderivative"] = @benchmarkable begin
            value_and_multiderivative($backend, $f, $x, $extras)
        end
        subgroup["value_and_multiderivative!"] = @benchmarkable begin
            value_and_multiderivative!($multider, $backend, $f, $x, $extras)
        end

        subgroup["multiderivative"] = @benchmarkable begin
            multiderivative($backend, $f, $x, $extras)
        end
        subgroup["multiderivative!"] = @benchmarkable begin
            multiderivative!($multider, $backend, $f, $x, $extras)
        end
    end

    if handles_mutation(backend)
        y = zeros(m)
        for extras in unique([nothing, prepare_multiderivative(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][backend_string(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_multiderivative!"] = @benchmarkable begin
                value_and_multiderivative!($y, $multider, $backend, $f, $x, $extras)
            end
        end
    end

    return nothing
end

function add_gradient_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert m == 1

    x = randn(n)
    grad = zeros(n)

    for extras in unique([nothing, prepare_gradient(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_gradient"] = @benchmarkable begin
            value_and_gradient($backend, $f, $x, $extras)
        end
        subgroup["value_and_gradient!"] = @benchmarkable begin
            value_and_gradient!($grad, $backend, $f, $x, $extras)
        end

        subgroup["gradient"] = @benchmarkable begin
            gradient($backend, $f, $x, $extras)
        end
        subgroup["gradient!"] = @benchmarkable begin
            gradient!($grad, $backend, $f, $x, $extras)
        end
    end

    return nothing
end

function add_jacobian_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = randn(n)
    jac = zeros(m, n)

    for extras in unique([nothing, prepare_jacobian(backend, f, x)])
        subgroup = suite[n][m]["allocating"][backend_string(backend)][pretty_extras(extras)]

        subgroup["value_and_jacobian"] = @benchmarkable begin
            value_and_jacobian($backend, $f, $x, $extras)
        end
        subgroup["value_and_jacobian!"] = @benchmarkable begin
            value_and_jacobian!($jac, $backend, $f, $x, $extras)
        end

        subgroup["jacobian"] = @benchmarkable begin
            jacobian($backend, $f, $x, $extras)
        end
        subgroup["jacobian!"] = @benchmarkable begin
            jacobian!($jac, $backend, $f, $x, $extras)
        end
    end

    if handles_mutation(backend)
        y = zeros(m)
        for extras in unique([nothing, prepare_jacobian(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][backend_string(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_jacobian!"] = @benchmarkable begin
                value_and_jacobian!($y, $jac, $backend, $f, $x, $extras)
            end
        end
    end

    return nothing
end

## Functions

struct Layer{W<:Union{Number,AbstractArray},B<:Union{Number,AbstractArray},S<:Function}
    w::W
    b::B
    σ::S
end

function (l::Layer{<:Number,<:Number})(x::Number)::Number
    return l.σ(l.w * x + l.b)
end

function (l::Layer{<:AbstractVector,<:AbstractVector})(x::Number)::AbstractVector
    return l.σ.(l.w .* x .+ l.b)
end

function (l!::Layer{<:AbstractVector,<:AbstractVector})(
    y::AbstractVector, x::Number
)::Nothing
    y .= l!.σ.(l!.w .* x .+ l!.b)
    return nothing
end

function (l::Layer{<:AbstractVector,<:Number})(x::AbstractVector)::Number
    return l.σ(dot(l.w, x) + l.b)
end

function (l::Layer{<:AbstractMatrix,<:AbstractVector})(x::AbstractVector)::AbstractVector
    return l.σ.(l.w * x .+ l.b)
end

function (l!::Layer{<:AbstractMatrix,<:AbstractVector})(
    y::AbstractVector, x::AbstractVector
)::Nothing
    mul!(y, l!.w, x)
    y .= l!.σ.(y .+ l!.b)
    return nothing
end

## Suite

function make_suite(;
    backends,
    included::Vector{Symbol}=[
        :pushforward, :pullback, :derivative, :multiderivative, :gradient, :jacobian
    ],
)
    SUITE = BenchmarkGroup()

    ### Scalar to scalar
    scalar_to_scalar = Layer(randn(), randn(), tanh)

    for backend in backends
        if :derivative in included
            add_derivative_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
        end
        if :pushforward in included
            add_pushforward_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
        end
        if :pullback in included
            add_pullback_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
        end
    end

    ### Scalar to vector
    for m in [10]
        scalar_to_vector = Layer(randn(m), randn(m), tanh)

        for backend in backends
            if :multiderivative in included
                add_multiderivative_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
            end
            if :pushforward in included
                add_pushforward_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
            end
            if :pullback in included
                add_pullback_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
            end
        end
    end

    ### Vector to scalar
    for n in [10]
        vector_to_scalar = Layer(randn(n), randn(), tanh)

        for backend in backends
            if :gradient in included
                add_gradient_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
            end
            if :pushforward in included
                add_pushforward_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
            end
            if :pullback in included
                add_pullback_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
            end
        end
    end

    ### Vector to vector
    for (n, m) in [(10, 10)]
        vector_to_vector = Layer(randn(m, n), randn(m), tanh)

        for backend in backends
            if :jacobian in included
                add_jacobian_benchmarks!(SUITE, backend, vector_to_vector, n, m)
            end
            if :pushforward in included
                add_pushforward_benchmarks!(SUITE, backend, vector_to_vector, n, m)
            end
            if :pullback in included
                add_pullback_benchmarks!(SUITE, backend, vector_to_vector, n, m)
            end
        end
    end

    return SUITE
end

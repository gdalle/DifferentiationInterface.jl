using DifferentiationInterface
using BenchmarkTools

function add_pushforward_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? randn() : randn(n)
    dy = m == 1 ? 0.0 : zeros(m)

    if autodiff_mode(backend) != :forward || !handles_types(backend, typeof(x), typeof(dy))
        return nothing
    end

    suite["value_and_pushforward"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_pushforward($backend, $f, $x, $dx)
    end
    suite["value_and_pushforward!"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_pushforward!($dy, $backend, $f, $x, $dx)
    end

    suite["pushforward"][(n, m)][string(backend)] = @benchmarkable begin
        pushforward($backend, $f, $x, $dx)
    end
    suite["pushforward!"][(n, m)][string(backend)] = @benchmarkable begin
        pushforward!($dy, $backend, $f, $x, $dx)
    end

    return nothing
end

function add_pullback_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? 0.0 : zeros(n)
    dy = m == 1 ? randn() : randn(m)

    if autodiff_mode(backend) != :reverse || !handles_types(backend, typeof(x), typeof(dy))
        return nothing
    end

    suite["value_and_pullback"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_pullback($backend, $f, $x, $dy)
    end
    suite["value_and_pullback!"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_pullback!($dx, $backend, $f, $x, $dy)
    end

    suite["pullback"][(n, m)][string(backend)] = @benchmarkable begin
        pullback($backend, $f, $x, $dy)
    end
    suite["pullback!"][(n, m)][string(backend)] = @benchmarkable begin
        pullback!($dx, $backend, $f, $x, $dy)
    end

    return nothing
end

function add_derivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    @assert n == m == 1
    if !handles_types(backend, Number, Number)
        return nothing
    end

    x = randn()

    suite["value_and_derivative"][(1, 1)][string(backend)] = @benchmarkable begin
        value_and_derivative($backend, $f, $x)
    end

    suite["derivative"][(1, 1)][string(backend)] = @benchmarkable begin
        derivative($backend, $f, $x)
    end

    return nothing
end

function add_multiderivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    @assert n == 1
    if !handles_types(backend, Number, Vector)
        return nothing
    end

    x = randn()
    multider = zeros(m)

    suite["value_and_multiderivative"][(1, m)][string(backend)] = @benchmarkable begin
        value_and_multiderivative($backend, $f, $x)
    end
    suite["value_and_multiderivative!"][(1, m)][string(backend)] = @benchmarkable begin
        value_and_multiderivative!($multider, $backend, $f, $x)
    end

    suite["multiderivative"][(1, m)][string(backend)] = @benchmarkable begin
        multiderivative($backend, $f, $x)
    end
    suite["multiderivative!"][(1, m)][string(backend)] = @benchmarkable begin
        multiderivative!($multider, $backend, $f, $x)
    end

    return nothing
end

function add_gradient_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    @assert m == 1
    if !handles_types(backend, Vector, Number)
        return nothing
    end

    x = randn(n)
    grad = zeros(n)

    suite["value_and_gradient"][(n, 1)][string(backend)] = @benchmarkable begin
        value_and_gradient($backend, $f, $x)
    end
    suite["value_and_gradient!"][(n, 1)][string(backend)] = @benchmarkable begin
        value_and_gradient!($grad, $backend, $f, $x)
    end

    suite["gradient"][(n, 1)][string(backend)] = @benchmarkable begin
        gradient($backend, $f, $x)
    end
    suite["gradient!"][(n, 1)][string(backend)] = @benchmarkable begin
        gradient!($grad, $backend, $f, $x)
    end

    return nothing
end

function add_jacobian_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractBackend, f::F, n::Integer, m::Integer
) where {F}
    if !handles_types(backend, Vector, Vector)
        return nothing
    end

    x = randn(n)
    jac = zeros(m, n)

    suite["value_and_jacobian"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_jacobian($backend, $f, $x)
    end
    suite["value_and_jacobian!"][(n, m)][string(backend)] = @benchmarkable begin
        value_and_jacobian!($jac, $backend, $f, $x)
    end

    suite["jacobian"][(n, m)][string(backend)] = @benchmarkable begin
        jacobian($backend, $f, $x)
    end
    suite["jacobian!"][(n, m)][string(backend)] = @benchmarkable begin
        jacobian!($jac, $backend, $f, $x)
    end

    return nothing
end

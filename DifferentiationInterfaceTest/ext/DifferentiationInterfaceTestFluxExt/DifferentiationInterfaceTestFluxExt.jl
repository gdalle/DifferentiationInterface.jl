module DifferentiationInterfaceTestFluxExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDifferences: FiniteDifferences
using Flux
using Functors
using LinearAlgebra
using Random: AbstractRNG, default_rng

#=
Relevant discussions:

- https://github.com/gdalle/DifferentiationInterface.jl/issues/105
- https://github.com/gdalle/DifferentiationInterface.jl/issues/343
- https://github.com/FluxML/Flux.jl/issues/2469
=#

function gradient_finite_differences(loss, model)
    v, re = Flux.destructure(model)
    fdm = FiniteDifferences.central_fdm(5, 1)
    gs = FiniteDifferences.grad(fdm, loss ∘ re, f64(v))
    return re(only(gs))
end

function DIT.flux_isequal(a, b)
    return all(isequal.(fleaves(a), fleaves(b)))
end

function DIT.flux_isapprox(a, b; atol, rtol)
    isapprox_results = fmapstructure_with_path(a, b) do kp, x, y
        if :state in kp  # ignore RNN and LSTM state
            return true
        else
            if x isa AbstractArray{<:Number}
                return isapprox(x, y; atol, rtol)
            else  # ignore non-arrays
                return true
            end
        end
    end
    return all(fleaves(isapprox_results))
end

struct SquareLossOnInput{X}
    x::X
end

struct SquareLossOnInputIterated{X}
    x::X
end

function (sqli::SquareLossOnInput)(model)
    Flux.reset!(model)
    return sum(abs2, model(sqli.x))
end

function (sqlii::SquareLossOnInputIterated)(model)
    Flux.reset!(model)
    x = copy(sqlii.x)
    for _ in 1:3
        x = model(x)
    end
    return sum(abs2, x)
end

struct SimpleDense{W,B,F}
    w::W
    b::B
    σ::F
end

(m::SimpleDense)(x) = m.σ.(m.w * x .+ m.b)

@functor SimpleDense

function DIT.flux_scenarios(rng::AbstractRNG=default_rng())
    scens = Scenario[]

    # Simple dense

    d_in, d_out = 4, 2
    w = randn(rng, d_out, d_in)
    b = randn(rng, d_out)
    model = SimpleDense(w, b, Flux.σ)

    x = randn(rng, d_in)
    loss = SquareLossOnInput(x)
    l = loss(model)
    g = gradient_finite_differences(loss, model)

    scen = GradientScenario(loss; x=model, y=l, grad=g, nb_args=1, place=:outofplace)
    push!(scens, scen)

    # Layers

    models_and_xs = [
        (Dense(2, 4), randn(rng, Float32, 2)),
        (Chain(Dense(2, 4, relu), Dense(4, 3)), randn(rng, Float32, 2)),
        (f64(Chain(Dense(2, 4), Dense(4, 2))), randn(Float64, 2, 1)),
        (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(rng, Float32, 2)),
        (Conv((3, 3), 2 => 3), randn(rng, Float32, 3, 3, 2, 1)),
        (
            Chain(Conv((3, 3), 2 => 3, relu), Conv((3, 3), 3 => 1, relu)),
            rand(rng, Float32, 5, 5, 2, 1),
        ),
        (
            Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())),
            rand(rng, Float32, 5, 5, 2, 2),
        ),
        (Maxout(() -> Dense(5 => 4, tanh), 3), randn(rng, Float32, 5, 1)),
        (RNN(3 => 2), randn(rng, Float32, 3, 2)),
        (Chain(RNN(3 => 4), RNN(4 => 3)), randn(rng, Float32, 3, 2)),
        (LSTM(3 => 5), randn(rng, Float32, 3, 2)),
        (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(rng, Float32, 3, 2)),
        (SkipConnection(Dense(2 => 2), vcat), randn(rng, Float32, 2, 3)),
        (Flux.Bilinear((2, 2) => 3), randn(rng, Float32, 2, 1)),
        (GRU(3 => 5), randn(rng, Float32, 3, 10)),
        (ConvTranspose((3, 3), 3 => 2; stride=2), rand(rng, Float32, 5, 5, 3, 1)),
    ]

    for (model, x) in models_and_xs
        Flux.trainmode!(model)
        loss = SquareLossOnInput(x)
        l = loss(model)
        g = gradient_finite_differences(loss, model)
        scen = GradientScenario(loss; x=model, y=l, grad=g, nb_args=1, place=:outofplace)
        push!(scens, scen)
    end

    # Recurrence

    recurrent_models_and_xs = [
        (RNN(3 => 3), randn(rng, Float32, 3, 2)),
        (LSTM(3 => 3), randn(rng, Float32, 3, 2)),
    ]

    for (model, x) in recurrent_models_and_xs
        Flux.trainmode!(model)
        loss = SquareLossOnInputIterated(x)
        l = loss(model)
        g = gradient_finite_differences(loss, model)
        scen = GradientScenario(loss; x=model, y=l, grad=g, nb_args=1, place=:outofplace)
        push!(scens, scen)
    end

    return scens
end

end

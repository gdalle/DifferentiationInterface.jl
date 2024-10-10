module DifferentiationInterfaceTestFluxExt

using DifferentiationInterface
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

function gradient_finite_differences(loss, model, x)
    v, re = Flux.destructure(model)
    fdm = FiniteDifferences.central_fdm(5, 1)
    gs = FiniteDifferences.grad(fdm, model -> loss(re(model), x), f64(v))
    return re(only(gs))
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

function square_loss(model, x)
    Flux.reset!(model)
    return sum(abs2, model(x))
end

function square_loss_iterated(model, x)
    Flux.reset!(model)
    y = copy(x)
    for _ in 1:3
        y = model(y)
    end
    return sum(abs2, y)
end

struct SimpleDense{W,B,F}
    w::W
    b::B
    σ::F
end

(m::SimpleDense)(x) = m.σ.(m.w * x .+ m.b)

@functor SimpleDense

function DIT.flux_scenarios(rng::AbstractRNG=default_rng())
    init = Flux.glorot_uniform(rng)

    scens = Scenario[]

    # Simple dense

    d_in, d_out = 4, 2
    w = randn(rng, d_out, d_in)
    b = randn(rng, d_out)
    model = SimpleDense(w, b, Flux.σ)

    x = randn(rng, d_in)
    g = gradient_finite_differences(square_loss, model, x)

    scen = Scenario{:gradient,:out}(square_loss, model; contexts=(Constant(x),), res1=g)
    push!(scens, scen)

    # Layers

    models_and_xs = [
        #! format: off
        (
            Dense(2, 4; init),
            randn(rng, Float32, 2)
        ),
        (
            Chain(Dense(2, 4, relu; init), Dense(4, 3; init)),
            randn(rng, Float32, 2)),
        (
            f64(Chain(Dense(2, 4; init), Dense(4, 2; init))),
            randn(rng, Float64, 2, 1)),
        (
            Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2),
            randn(rng, Float32, 2)),
        (
            Conv((3, 3), 2 => 3; init),
            randn(rng, Float32, 3, 3, 2, 1)),
        (
            Chain(Conv((3, 3), 2 => 3, relu; init), Conv((3, 3), 3 => 1, relu; init)),
            rand(rng, Float32, 5, 5, 2, 1),
        ),
        (
            Chain(Conv((4, 4), 2 => 2; pad=SamePad(), init), MeanPool((5, 5); pad=SamePad())),
            rand(rng, Float32, 5, 5, 2, 2),
        ),
        (
            Maxout(() -> Dense(5 => 4, tanh; init), 3),
            randn(rng, Float32, 5, 1)
        ),
        (
            RNN(3 => 2; init),
            randn(rng, Float32, 3, 2)
        ),
        (
            Chain(RNN(3 => 4; init), RNN(4 => 3; init)),
            randn(rng, Float32, 3, 2)
        ),
        (
            LSTM(3 => 5; init),
            randn(rng, Float32, 3, 2)
        ),
        (
            Chain(LSTM(3 => 5; init), LSTM(5 => 3; init)),
            randn(rng, Float32, 3, 2)
        ),
        (
            SkipConnection(Dense(2 => 2; init), vcat),
            randn(rng, Float32, 2, 3)
        ),
        (
            Flux.Bilinear((2, 2) => 3; init),
            randn(rng, Float32, 2, 1)
        ),
        (
            GRU(3 => 5; init),
            randn(rng, Float32, 3, 10)
        ),
        (
            ConvTranspose((3, 3), 3 => 2; stride=2, init),
            rand(rng, Float32, 5, 5, 3, 1)
        ),
    #! format: on
    ]

    for (model, x) in models_and_xs
        Flux.trainmode!(model)
        g = gradient_finite_differences(square_loss, model, x)
        scen = Scenario{:gradient,:out}(square_loss, model; contexts=(Constant(x),), res1=g)
        push!(scens, scen)
    end

    # Recurrence

    recurrent_models_and_xs = [
        #! format: off
        (
            RNN(3 => 3; init),
            randn(rng, Float32, 3, 2)
        ),
        (
            LSTM(3 => 3; init),
            randn(rng, Float32, 3, 2)
        ),
        #! format: on
    ]

    for (model, x) in recurrent_models_and_xs
        Flux.trainmode!(model)
        g = gradient_finite_differences(square_loss, model, x)
        scen = Scenario{:gradient,:out}(
            square_loss_iterated, model; contexts=(Constant(x),), res1=g
        )
        # TODO: figure out why these tests are broken
        # push!(scens, scen)
    end

    return scens
end

end
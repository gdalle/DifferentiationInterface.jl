module DifferentiationInterfaceTestFluxExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDifferences: FiniteDifferences
using Flux
using Functors
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
    gs = FiniteDifferences.grad(fdm, loss ∘ re, v)
    return re(only(gs))
end

function DIT.flux_isequal(a, b)
    return all(isequal.(fleaves(a), fleaves(b)))
end

function DIT.flux_isapprox(a, b; atol, rtol)
    isapprox_results = fmap_with_path(a, b) do kp, x, y
        :state ∈ kp && return nothing # ignore RNN and LSTM state
        if x isa AbstractArray{<:Number} && !isapprox(x, y; atol, rtol)
            return false
        else
            return true
        end
    end
    return all(isapprox_results)
end

struct SquareLossOnInput{X}
    x::X
end

function (sqli::SquareLossOnInput)(model)
    Flux.reset!(model)
    return sum(abs2, model(sqli.x))
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

    return scens
end

end

@kwdef struct Layer{W,B,A}
    w::W
    b::B
    σ::A = nothing
end

@functor Layer

(l::Layer{<:Number,<:Number,<:Nothing})(x::Number) = l.w * x + l.b
(l::Layer{<:Number,<:Number})(x::Number) = l.σ(l.w * x + l.b)

(l::Layer{<:AbstractMatrix,<:AbstractVector,<:Nothing})(x::AbstractVector) = l.w * x + l.b
(l::Layer{<:AbstractMatrix,<:AbstractVector})(x::AbstractVector) = l.σ.(l.w * x + l.b)

call_layer(l::Layer{<:Number,<:Number}) = l(3.0)
call_layer(l::Layer{<:AbstractMatrix,<:AbstractVector}) = l(3 * ones(size(l.w, 2)))

sum_call_layer(l) = sum(call_layer(l))

nested_norm(x::Number) = abs2(x)
nested_norm(x::AbstractArray) = sum(abs2, x)
nested_norm(x) = sum(nested_norm, fleaves(x))

function make_complicated(x::Number)
    return (a=[2x, 3x], b=([exp(x);;],))
end

function make_complicated_with_immutables(x::Number)
    return (a=[2x, 3x], b=([exp(x);;],), c=sin(x))
end

function nested_scenarios(; immutables=true)
    scenarios_without_immutables = [
        Scenario(make_complicated, 2.0),
        Scenario(nested_norm; x=make_complicated(2.0)),
        Scenario(sum_call_layer; x=Layer(; w=rand(2, 3), b=rand(2))),
        Scenario(sum_call_layer; x=Layer(; w=rand(2, 3), b=rand(2), σ=tanh)),
    ]
    scenarios_with_immutables = [
        Scenario(make_complicated_with_immutables, 2.0),
        Scenario(nested_norm; x=make_complicated_with_immutables(2.0)),
        Scenario(call_layer; x=Layer(; w=2.0, b=4.0)),
        Scenario(call_layer; x=Layer(; w=2.0, b=4.0, σ=tanh)),
    ]
    if immutables
        return vcat(scenarios_with_immutables, scenarios_without_immutables)
    else
        return scenarios_without_immutables
    end
end

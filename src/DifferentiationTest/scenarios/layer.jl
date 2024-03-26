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

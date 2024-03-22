# Getting started

## [Operators](@id operators)

Depending on the type of input and output, differentiation operators can have various names.
Most backends have custom implementations, which we reuse if possible.

We choose the following terminology for the high-level operators we provide:

| input  `x`      | output   `y`    | operator   | result type                     |
| --------------- | --------------- | ---------- | ------------------------------- |
| `Number`        | `Any`           | derivative | same as output                  |
| `Any`           | `Number`        | gradient   | same as input                   |
| `AbstractArray` | `AbstractArray` | Jacobian   | matrix `(length(y), length(x))` |

They are all based on the following low-level operators:

- pushforward, to propagate input tangents
- pullback, to propagate output cotangents

!!! tip
    See the documentation of [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) for details on these concepts.

## Variants

Several variants of each operator are defined:

| operator    | out-of-place                    | in-place                         |
| :---------- | :------------------------------ | :------------------------------- |
| derivative  | [`value_and_derivative`](@ref)  | [`value_and_derivative!`](@ref)  |
| gradient    | [`value_and_gradient`](@ref)    | [`value_and_gradient!`](@ref)    |
| Jacobian    | [`value_and_jacobian`](@ref)    | [`value_and_jacobian!`](@ref)    |
| pushforward | [`value_and_pushforward`](@ref) | [`value_and_pushforward!`](@ref) |
| pullback    | [`value_and_pullback`](@ref)    | [`value_and_pullback!`](@ref)    |

## Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.
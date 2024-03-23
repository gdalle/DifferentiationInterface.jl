# Getting started

## [Operators](@id operators)

Depending on the type of input and output, differentiation operators can have various names.
Most backends have custom implementations, which we reuse if possible.

We choose the following terminology for the high-level operators we provide:

| operator   | input  `x`      | output   `y`    | result type      | result shape             |
| ---------- | --------------- | --------------- | ---------------- | ------------------------ |
| derivative | `Number`        | `Any`           | same as `y`      | `size(y)`                |
| gradient   | `Any`           | `Number`        | same as `x`      | `size(x)`                |
| Jacobian   | `AbstractArray` | `AbstractArray` | `AbstractMatrix` | `(length(y), length(x))` |

They are all based on the following low-level operators:

- pushforward (or JVP), to propagate input tangents
- pullback (or VJP), to backpropagate output cotangents

!!! tip
    See the book [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) for details on these concepts.

## Variants

Several variants of each operator are defined:

| operator    | out-of-place                    | in-place (if possible)            |
| :---------- | :------------------------------ | :-------------------------------- |
| derivative  | [`value_and_derivative`](@ref)  | [`value_and_derivative!!`](@ref)  |
| gradient    | [`value_and_gradient`](@ref)    | [`value_and_gradient!!`](@ref)    |
| Jacobian    | [`value_and_jacobian`](@ref)    | [`value_and_jacobian!!`](@ref)    |
| pushforward | [`value_and_pushforward`](@ref) | [`value_and_pushforward!!`](@ref) |
| pullback    | [`value_and_pullback`](@ref)    | [`value_and_pullback!!`](@ref)    |

!!! warning
    The "bang-bang" syntactic convention `!!` signals that some of the arguments _can_ be mutated, but they do not _have to be_.
    Users should not rely on mutation, but instead recover the function output and work from there.
    ```julia
        y, grad = value_and_gradient!!(f, grad, backend, x)  # good
        value_and_gradient!!(f, grad, backend, x)  # bad
    ```

## Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.
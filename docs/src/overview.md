# Overview

## Operators

Depending on the type of input and output, differentiation operators can have various names.
Most backends have custom implementations, which we reuse if possible.

We choose the following terminology for the high-level operators we provide:

| operator             | input  `x`      | output   `y`    | result type      | result shape             |
| -------------------- | --------------- | --------------- | ---------------- | ------------------------ |
| [`derivative`](@ref) | `Number`        | `Any`           | same as `y`      | `size(y)`                |
| [`gradient`](@ref)   | `Any`           | `Number`        | same as `x`      | `size(x)`                |
| [`jacobian`](@ref)   | `AbstractArray` | `AbstractArray` | `AbstractMatrix` | `(length(y), length(x))` |

They are all based on the following low-level operators:

- [`pushforward`](@ref) (or JVP), to propagate input tangents
- [`pullback`](@ref) (or VJP), to backpropagate output cotangents

!!! tip
    See the book [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) for details on these concepts.

## Variants

Several variants of each operator are defined:

| out-of-place          | in-place (or not)       | out-of-place + primal           | in-place (or not) + primal        |
| :-------------------- | :---------------------- | :------------------------------ | :-------------------------------- |
| [`derivative`](@ref)  | [`derivative!!`](@ref)  | [`value_and_derivative`](@ref)  | [`value_and_derivative!!`](@ref)  |
| [`gradient`](@ref)    | [`gradient!!`](@ref)    | [`value_and_gradient`](@ref)    | [`value_and_gradient!!`](@ref)    |
| [`jacobian`](@ref)    | [`jacobian!!`](@ref)    | [`value_and_jacobian`](@ref)    | [`value_and_jacobian!!`](@ref)    |
| [`pushforward`](@ref) | [`pushforward!!`](@ref) | [`value_and_pushforward`](@ref) | [`value_and_pushforward!!`](@ref) |
| [`pullback`](@ref)    | [`pullback!!`](@ref)    | [`value_and_pullback`](@ref)    | [`value_and_pullback!!`](@ref)    |

!!! warning
    The "bang-bang" syntactic convention `!!` signals that some of the arguments _can_ be mutated, but they do not _have to be_.
    Such arguments will always be part of the return, so that one can simply reuse the operator's output and forget its input.

    In other words, this is good:
    ```julia
    grad = gradient!!(f, grad, backend, x)  # do this
    ```
    On the other hand, this is bad, because if `grad` has not been mutated, you will get wrong results:
    ```julia
    gradient!!(f, grad, backend, x)  # don't do this
    ```

## Second order

Second-order differentiation is also supported, with the following operators:

| operator                    | input  `x`      | output   `y` | result type      | result shape             |
| --------------------------- | --------------- | ------------ | ---------------- | ------------------------ |
| [`second_derivative`](@ref) | `Number`        | `Any`        | same as `y`      | `size(y)`                |
| [`hvp`](@ref)               | `Any`           | `Number`     | same as `x`      | `size(x)`                |
| [`hessian`](@ref)           | `AbstractArray` | `Number`     | `AbstractMatrix` | `(length(x), length(x))` |

!!! danger
    This is an experimental functionality, use at your own risk.

## Preparation

In many cases, automatic differentiation can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided.
This is a backend-specific procedure, but we expose a common syntax to achieve it.

| operator            | preparation function                |
| :------------------ | :---------------------------------- |
| `derivative`        | [`prepare_derivative`](@ref)        |
| `gradient`          | [`prepare_gradient`](@ref)          |
| `jacobian`          | [`prepare_jacobian`](@ref)          |
| `second_derivative` | [`prepare_second_derivative`](@ref) |
| `hessian`           | [`prepare_hessian`](@ref)           |
| `pushforward`       | [`prepare_pushforward`](@ref)       |
| `pullback`          | [`prepare_pullback`](@ref)          |
| `hvp`               | [`prepare_hvp`](@ref)               |

If you run `prepare_operator(backend, f, x)`, it will create an object called `extras` containing the necessary information to speed up `operator` and its variants.
This information is specific to `backend` and `f`, as well as the _type and size_ of the input `x`, but it should work with different _values_ of `x`.

You can then call `operator(backend, f, similar_x, extras)`, which should be faster than `operator(backend, f, similar_x)`.
This is especially worth it if you plan to call `operator` several times in similar settings: you can think of it as a warm up.

By default, all the preparation functions return `nothing`.
We do not make any guarantees on their implementation for each backend, or on the performance gains that can be expected.

!!! warning
    We haven't fully figured out what must happen when an `extras` object is prepared for a specific operator but then given to a lower-level one (i.e. prepare it for `jacobian` but then give it to `pushforward` inside `jacobian`).

## Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.

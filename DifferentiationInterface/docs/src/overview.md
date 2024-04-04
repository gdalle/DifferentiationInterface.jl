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
    # work with grad_in
    grad_out = gradient!!(f, grad_in, backend, x)
    # work with grad_out: OK
    ```
    On the other hand, this is bad, because if `grad_in` has not been mutated, you will forget the results:
    ```julia
    # work with grad_in
    gradient!!(f, grad_in, backend, x)
    # mistakenly keep working with grad_in: NOT OK
    ```
    Note that we don't guarantee `grad_out` will have the same type as `grad_in`.
    Its type can even depend on the choice of backend.

## Second order

Second-order differentiation is also supported.
You can either pick a single backend to do all the work, or combine an "outer" backend with an "inner" backend using the [`SecondOrder`](@ref) struct, like so: `SecondOrder(outer, inner)`.

The available operators are similar to first-order ones:

| operator                    | input  `x`      | output   `y` | result type      | result shape             |
| --------------------------- | --------------- | ------------ | ---------------- | ------------------------ |
| [`second_derivative`](@ref) | `Number`        | `Any`        | same as `y`      | `size(y)`                |
| [`hvp`](@ref)               | `Any`           | `Number`     | same as `x`      | `size(x)`                |
| [`hessian`](@ref)           | `AbstractArray` | `Number`     | `AbstractMatrix` | `(length(x), length(x))` |

We only define two variants for now:

| out-of-place                | in-place (or not)             |
| --------------------------- | ----------------------------- |
| [`second_derivative`](@ref) | [`second_derivative!!`](@ref) |
| [`hvp`](@ref)               | [`hvp!!`](@ref)               |
| [`hessian`](@ref)           | [`hessian!!`](@ref)           |

!!! danger
    Second-order differentiation is still experimental, use at your own risk.

## Preparation

In many cases, AD can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided.
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
    For `SecondOrder` backends, the inner differentiation cannot be prepared at the moment, only the outer one is.

## FAQ

### Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.

### Sparsity

If you need to work with sparse Jacobians, you can pick one of the [sparse backends](@ref Sparse) from [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
The sparsity pattern is computed automatically with [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) during the preparation step.

If you need to work with sparse Hessians, you can use a sparse backend as the _outer_ backend of a `SecondOrder`.
This means the Hessian is obtained as the sparse Jacobian of the gradient.

!!! danger
    Sparsity support is still experimental, use at your own risk.

### Split reverse mode

Many reverse mode AD backends expose a "split" option, which runs only the forward sweep, and encapsulates the reverse sweep in a closure.
We make this available for everyone with the following operators:

| out-of-place                       | in-place (or not)                    |
| ---------------------------------- | ------------------------------------ |
| [`value_and_pullback_split`](@ref) | [`value_and_pullback!!_split`](@ref) |

!!! danger
    Split reverse mode is still experimental, use at your own risk.
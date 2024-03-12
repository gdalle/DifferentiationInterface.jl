# Getting started

## [Operators](@id operators)

Depending on the type of input and output, differentiation operators can have various names.
We choose the following terminology for the utilities we provide:

|                  | **scalar output**   | **array output**  |
| ---------------- | ------------------- | ----------------- |
| **array input**  | `gradient`          | `jacobian`        |
| **scalar input** | `derivative`        | `multiderivative` |

Most backends have custom implementations for all of these, which we reuse whenever possible.

### Variants

Whenever it makes sense, four variants of the same operator are defined:

| **Operator**      | **non-mutating**          | **mutating**                 | **non-mutating with primal**        | **mutating with primal**             |
|:------------------|:--------------------------|:-----------------------------|:------------------------------------|:-------------------------------------|
| Gradient          | [`gradient`](@ref)        | [`gradient!`](@ref)          | [`value_and_gradient`](@ref)        | [`value_and_gradient!`](@ref)        |
| Jacobian          | [`jacobian`](@ref)        | [`jacobian!`](@ref)          | [`value_and_jacobian`](@ref)        | [`value_and_jacobian!`](@ref)        |
| Multiderivative   | [`multiderivative`](@ref) | [`multiderivative!`](@ref)   | [`value_and_multiderivative`](@ref) | [`value_and_multiderivative!`](@ref) |
| Derivative        | [`derivative`](@ref)      | N/A                          | [`value_and_derivative`](@ref)      | N/A                                  | 
| Pullback (VJP)    | [`pullback`](@ref)        | [`pullback!`](@ref)          | [`value_and_pullback`](@ref)        | [`value_and_pullback!`](@ref)        |
| Pushforward (JVP) | [`pushforward`](@ref)     | [`pushforward!`](@ref)       | [`value_and_pushforward`](@ref)     | [`value_and_pushforward!`](@ref)     |

Note that scalar outputs can't be mutated, which is why `derivative` doesn't have mutating variants.

## Preparation

In many cases, automatic differentiation can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided.
This is a backend-specific procedure, but we expose a common syntax to achieve it.

| **Operator**      | **preparation function**          |
|:------------------|:----------------------------------|
| Gradient          | [`prepare_gradient`](@ref)        |
| Jacobian          | [`prepare_jacobian`](@ref)        |
| Multiderivative   | [`prepare_multiderivative`](@ref) |
| Derivative        | [`prepare_derivative`](@ref)      |
| Pullback (VJP)    | [`prepare_pullback`](@ref)        |
| Pushforward (JVP) | [`prepare_pushforward`](@ref)     |

If you run `prepare_operator(backend, f, x)`, it will create an object called `extras` containing the necessary information to speed up `operator` and its variants.
This information is specific to `backend` and `f`, as well as the _type and size_ of the input `x`, but it should work with different _values_ of `x`.
You can them call `operator(backend, f, similar_x, extras)`, which should be faster than `operator(backend, f, similar_x)`.
This is especially worth it if you plan to call `operator` several times in similar settings: you can think of it as a warm up.

By default, all the preparation functions return `nothing`.
We do not make any guarantees on their implementation for each backend, or on the performance gains that can be expected.

# Getting started

## [Operators](@id operators)

Depending on the type of input and output, differentiation operators can have various names.
Most backends have custom implementations, which we reuse if possible.

We choose the following terminology for the operators we provide:

### First order differentiation

|                  | **scalar output** | **array output**  |
| ---------------- | ----------------- | ----------------- |
| **scalar input** | `derivative`      | `multiderivative` |
| **array input**  | `gradient`        | `jacobian`        |

### Second order differentiation

|                  | **scalar output**   | **array output** |
| ---------------- | ------------------- | ---------------- |
| **scalar input** | `second_derivative` | not implemented  |
| **array input**  | `hessian`           | not implemented  |

## Variants

Whenever it makes sense, four variants of the same operator are defined:

| **Operator**      | **allocating**              | **mutating**               | **allocating with primal**                       | **mutating with primal**              |
| :---------------- | :-------------------------- | :------------------------- | :----------------------------------------------- | :------------------------------------ |
| Derivative        | [`derivative`](@ref)        | N/A                        | [`value_and_derivative`](@ref)                   | N/A                                   |
| Multiderivative   | [`multiderivative`](@ref)   | [`multiderivative!`](@ref) | [`value_and_multiderivative`](@ref)              | [`value_and_multiderivative!`](@ref)  |
| Gradient          | [`gradient`](@ref)          | [`gradient!`](@ref)        | [`value_and_gradient`](@ref)                     | [`value_and_gradient!`](@ref)         |
| Jacobian          | [`jacobian`](@ref)          | [`jacobian!`](@ref)        | [`value_and_jacobian`](@ref)                     | [`value_and_jacobian!`](@ref)         |
| Second derivative | [`second_derivative`](@ref) | N/A                        | [`value_derivative_and_second_derivative`](@ref) | N/A                                   |
| Hessian           | [`hessian`](@ref)           | [`hessian!`](@ref)         | [`value_gradient_and_hessian`](@ref)             | [`value_gradient_and_hessian!`](@ref) |

Note that scalar outputs can't be mutated, which is why `derivative` and `second_derivative` do not have mutating variants.

For advanced users, lower-level operators are also exposed:

| **Operator**                 | **allocating**                   | **mutating**                      | **allocating with primal**                    | **mutating with primal**                       |
| :--------------------------- | :------------------------------- | :-------------------------------- | :-------------------------------------------- | :--------------------------------------------- |
| Pushforward (JVP)            | [`pushforward`](@ref)            | [`pushforward!`](@ref)            | [`value_and_pushforward`](@ref)               | [`value_and_pushforward!`](@ref)               |
| Pullback (VJP)               | [`pullback`](@ref)               | [`pullback!`](@ref)               | [`value_and_pullback`](@ref)                  | [`value_and_pullback!`](@ref)                  |
| Hessian-vector product (HVP) | [`hessian_vector_product`](@ref) | [`hessian_vector_product!`](@ref) | [`gradient_and_hessian_vector_product`](@ref) | [`gradient_and_hessian_vector_product!`](@ref) |

## Preparation

In many cases, automatic differentiation can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided.
This is a backend-specific procedure, but we expose a common syntax to achieve it.

| **Operator**                 | **preparation function**                 |
| :--------------------------- | :--------------------------------------- |
| Derivative                   | [`prepare_derivative`](@ref)             |
| Multiderivative              | [`prepare_multiderivative`](@ref)        |
| Gradient                     | [`prepare_gradient`](@ref)               |
| Jacobian                     | [`prepare_jacobian`](@ref)               |
| Second derivative            | [`prepare_second_derivative`](@ref)      |
| Hessian                      | [`prepare_hessian`](@ref)                |
| Pushforward (JVP)            | [`prepare_pushforward`](@ref)            |
| Pullback (VJP)               | [`prepare_pullback`](@ref)               |
| Hessian-vector product (HVP) | [`prepare_hessian_vector_product`](@ref) |

If you run `prepare_operator(backend, f, x)`, it will create an object called `extras` containing the necessary information to speed up `operator` and its variants.
This information is specific to `backend` and `f`, as well as the _type and size_ of the input `x`, but it should work with different _values_ of `x`.

You can then call `operator(backend, f, similar_x, extras)`, which should be faster than `operator(backend, f, similar_x)`.
This is especially worth it if you plan to call `operator` several times in similar settings: you can think of it as a warm up.

By default, all the preparation functions return `nothing`.
We do not make any guarantees on their implementation for each backend, or on the performance gains that can be expected.

## Mutating functions

In addition to allocating functions `f(x) = y`, some backends also support mutating functions `f!(y, x) = nothing` whenever the output is an array.
Beware that the function `f!` must return `nothing`!.
Since `f!` operates in-place and the primal is computed every time, only four operators are defined:

| **Operator**      | **mutating with primal**             |
| :---------------- | :----------------------------------- |
| Multiderivative   | [`value_and_multiderivative!`](@ref) |
| Jacobian          | [`value_and_jacobian!`](@ref)        |
| Pushforward (JVP) | [`value_and_pushforward!`](@ref)     |
| Pullback (VJP)    | [`value_and_pullback!`](@ref)        |

Furthermore, the preparation function takes an additional argument: `prepare_operator(backend, f!, x, y)`.

Check out the list of [backends that support mutating functions](@ref backend_mutation_behavior).

## Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.
# [Overview](@id sec-overview)

## Operators

The name of the differentiation operators can vary depending on the type of the input `x` and the type of the output `y` of the function being differentiated.

We provide the following high-level operators:

| operator                    | order | input  `x`      | output   `y`                | operator result type      | operator result shape    |
| :-------------------------- | :---- | :-------------- | :-------------------------- | :------------------------ | :----------------------- |
| [`derivative`](@ref)        | 1     | `Number`        | `Number` or `AbstractArray` | same as `y`               | `size(y)`                |
| [`second_derivative`](@ref) | 2     | `Number`        | `Number` or `AbstractArray` | same as `y`               | `size(y)`                |
| [`gradient`](@ref)          | 1     | `AbstractArray` | `Number`                    | same as `x`               | `size(x)`                |
| [`hessian`](@ref)           | 2     | `AbstractArray` | `Number`                    | `AbstractMatrix`          | `(length(x), length(x))` |
| [`jacobian`](@ref)          | 1     | `AbstractArray` | `AbstractArray`             | `AbstractMatrix`          | `(length(y), length(x))` |

They can be derived from lower-level operators:

| operator                       | order | input  `x`      | output   `y` | seed `v` | operator result type | operator result shape |
| :----------------------------- | :---- | :-------------- | :----------- | :------- | :------------------- | :-------------------- |
| [`pushforward`](@ref) (or JVP) | 1     | `Any`           | `Any`        | `dx`     | same as `y`          | `size(y)`             |
| [`pullback`](@ref) (or VJP)    | 1     | `Any`           | `Any`        | `dy`     | same as `x`          | `size(x)`             |
| [`hvp`](@ref)                  | 2     | `AbstractArray` | `Number`     | `dx`     | same as `x`          | `size(x)`             |

Luckily, most backends have custom implementations, which we reuse if possible instead of relying on fallbacks.

!!! tip
    See the book [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) for details on these concepts.

## Variants

Several variants of each operator are defined:

| out-of-place                | in-place                     | out-of-place + primal           | in-place + primal                |
| :-------------------------- | :--------------------------- | :------------------------------ | :------------------------------- |
| [`derivative`](@ref)        | [`derivative!`](@ref)        | [`value_and_derivative`](@ref)  | [`value_and_derivative!`](@ref)  |
| [`second_derivative`](@ref) | [`second_derivative!`](@ref) | NA                              | NA                               |
| [`gradient`](@ref)          | [`gradient!`](@ref)          | [`value_and_gradient`](@ref)    | [`value_and_gradient!`](@ref)    |
| [`hessian`](@ref)           | [`hessian!`](@ref)           | NA                              | NA                               |
| [`jacobian`](@ref)          | [`jacobian!`](@ref)          | [`value_and_jacobian`](@ref)    | [`value_and_jacobian!`](@ref)    |
| [`pushforward`](@ref)       | [`pushforward!`](@ref)       | [`value_and_pushforward`](@ref) | [`value_and_pushforward!`](@ref) |
| [`pullback`](@ref)          | [`pullback!`](@ref)          | [`value_and_pullback`](@ref)    | [`value_and_pullback!`](@ref)    |
| [`hvp`](@ref)               | [`hvp!`](@ref)               | NA                              | NA                               |

## Mutation and signatures

In order to ensure symmetry between one-argument functions `f(x) = y` and two-argument functions `f!(y, x) = nothing`, we define the same operators for both cases.
However they have different signatures:

| signature  | out-of-place                                 | in-place                                              |
| :--------- | :------------------------------------------- | :---------------------------------------------------- |
| `f(x)`     | `operator(f,     backend, x, [v], [extras])` | `operator!(f,     result, backend, x, [v], [extras])` |
| `f!(y, x)` | `operator(f!, y, backend, x, [v], [extras])` | `operator!(f!, y, result, backend, x, [v], [extras])` |

!!! warning
    Our mutation convention is that all positional arguments between `f`/`f!` and `backend` are mutated (the `extras` as well, see below).
    This convention holds regardless of the bang `!` in the operator name, because we assume that a user passing a two-argument function `f!(y, x)` anticipates mutation anyway.
    Still, better be careful with two-argument functions, because every variant of the operator will mutate `y`... even if it does not have a `!` in its name (see the bottom left cell in the table).

## Preparation

In many cases, AD can be accelerated if the function has been run at least once (e.g. to create a config or record a tape) and if some cache objects are provided.
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

Unsurprisingly, preparation syntax depends on the number of arguments:

| signature  | preparation signature                      |
| :--------- | :----------------------------------------- |
| `f(x)`     | `prepare_operator(f,     backend, x, [v])` |
| `f!(y, x)` | `prepare_operator(f!, y, backend, x, [v])` |

The preparation `prepare_operator(f, backend, x)` will create an object called `extras` containing the necessary information to speed up `operator` and its variants.
This information is specific to `backend` and `f`, as well as the _type and size_ of the input `x` and the _control flow_ within the function, but it should work with different _values_ of `x`.

You can then call e.g. `operator(backend, f, x2, extras)`, which should be faster than `operator(f, backend, x2)`.
This is especially worth it if you plan to call `operator` several times in similar settings: you can think of it as a warm up.

!!! warning
    The `extras` object is nearly always mutated when given to an operator, even when said operator does not have a bang `!` in its name.

### Second order

We offer two ways to perform second-order differentiation (for [`second_derivative`](@ref), [`hvp`](@ref) and [`hessian`](@ref)):

- pick a single backend to do all the work
- combine an "outer" and "inner" backend within the [`SecondOrder`](@ref) struct: the inner backend will be called first, and the outer backend will differentiate the generated code

!!! warning
    There are many possible backend combinations, a lot of which will fail.
    At the moment, trial and error is your best friend.
    Usually, the most efficient approach for Hessians is forward-over-reverse, i.e. a forward-mode outer backend and a reverse-mode inner backend.

!!! warning
    Preparation does not yet work for the inner differentiation step of a `SecondOrder`, only the outer differentiation is prepared.

## Experimental

!!! danger
    Everything in this section is still experimental, use it at your own risk.

### Sparsity

[ADTypes.jl](https://github.com/SciML/ADTypes.jl) provides [`AutoSparse`](@ref) to accelerate the computation of sparse Jacobians and Hessians.
Just wrap it around any backend, with an appropriate choice of sparsity detector and coloring algorithm, and call `jacobian` or `hessian`: the result will be sparse.
See the [tutorial section on sparsity](@ref sparsity-tutorial) for details.

### Split reverse mode

Some reverse mode AD backends expose a "split" option, which runs only the forward sweep, and encapsulates the reverse sweep in a closure.
We make this available for all backends with the following operators:

| out-of-place                       | in-place                            |
| :--------------------------------- | :---------------------------------- |
| [`value_and_pullback_split`](@ref) | [`value_and_pullback!_split`](@ref) |

### Translation

The wrapper [`DifferentiateWith`](@ref) allows you to translate between AD backends.
It takes a function `f` and specifies that `f` should be differentiated with the backend of your choice, instead of whatever other backend the code is trying to use.
In other words, when someone tries to differentiate `dw = DifferentiateWith(f, backend1)` with `backend2`, then `backend1` steps in and `backend2` does nothing.
At the moment, `DifferentiateWith` only works when `backend2` supports [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).

## Going further

### Non-standard types

The package is thoroughly tested with inputs and outputs of the following types: `Float64`, `Vector{Float64}` and `Matrix{Float64}`.
We also expect it to work on most kinds of `Number` and `AbstractArray` variables.
Beyond that, you are in uncharted territory.
We voluntarily keep the type annotations minimal, so that passing more complex objects or custom structs _might work with some backends_, but we make no guarantees about that.

### Multiple inputs/outputs

Restricting the API to one input and one output has many coding advantages, but it is not very flexible.
If you need more than that, use [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) to wrap several objects inside a single `ComponentVector`.

### Batched evaluation

This is not supported at the moment, but we plan to allow several pushforward / pullback seeds at once (similar to the chunking in ForwardDiff.jl or the batches in Enzyme.jl).

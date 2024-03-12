# Design

The operators defined in this package are split into two main parts:

- the "utilities", which are sufficient for most users
- the "primitives", which are mostly relevant for experts or backend developers

## Utilities

Depending on the type of input and output, differentiation operators can have various names.
We choose the following terminology for the utilities we provide:

|                  | **scalar output** | **array output** |
| ---------------- | ----------------- | ---------------- |
| **scalar input** | derivative        | multiderivative  |
| **array input**  | gradient          | jacobian         |

Most backends have custom implementations for all of these, which we reuse whenever possible.

## Primitives

Every utility can also be implemented from either of these two primitives:

- the pushforward (in forward mode), computing a Jacobian-vector product
- the pullback (in reverse mode), computing a vector-Jacobian product

## Variants

Whenever it makes sense, four variants of the same operator are defined:

|                       | **mutating**                             | **non-mutating**               |
| --------------------- | ---------------------------------------- | ------------------------------ |
| **primal too**        | `value_and_something!(storage, args...)` | `value_and_something(args...)` |
| **differential only** | `something!(storage, args...)`           | `something(args...)`           |

Replace `something` with `derivative`, `multiderivative`, `gradient`, `jacobian`, `pushforward` or `pullback` to get the correct name.

## Preparation

In many cases, automatic differentiation can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided.
This is a backend-specific procedure, but we expose a common syntax to achieve it.

If you run `prepare_something(backend, f, x)`, it will create an object called `extras` containing the necessary information to speed up the `something` procedure and its variants.
You can them call `something(backend, f, x, extras)`, which should be faster than `something(backend, f, x)`.
This is especially worth it if you plan to call `something` several times in similar settings (same backend, same function).
You can think of it as a warm up.

By default, all the preparation functions return `nothing`.
We do not make any guarantees on their implementation for each backend, or on the performance gains that can be expected.

## Backend requirements

The only requirement for a backend is to implement either [`value_and_pushforward!`](@ref) or [`value_and_pullback!`](@ref), from which the rest of the operators can be deduced.
We provide a standard series of fallbacks, but we leave it to each backend to redefine as many of the utilities as necessary to achieve optimal performance.

Every backend we support corresponds to a package extension of DifferentiationInterface.jl (located in the `ext` subfolder).
Advanced users are welcome to code more backends and submit pull requests!

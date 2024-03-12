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

## Backend requirements

The only requirement for a backend is to implement either [`value_and_pushforward!`](@ref) or [`value_and_pullback!`](@ref), from which the rest of the operators can be deduced.
Further methods are only defined for performance purposes, if they don't exist then a fallback structure kicks in.

# For AD developers

## Backend requirements

Every [operator](@ref operators) can be implemented from either of these two primitives:

- the pushforward (in forward mode), computing a Jacobian-vector product
- the pullback (in reverse mode), computing a vector-Jacobian product

The only requirement for a backend is therefore to implement either [`value_and_pushforward!`](@ref) or [`value_and_pullback!`](@ref), from which the rest of the operators can be deduced.
We provide a standard series of fallbacks, but we leave it to each backend to redefine as many of the utilities as necessary to achieve optimal performance.

Every backend we support corresponds to a package extension of DifferentiationInterface.jl (located in the `ext` subfolder).
Advanced users are welcome to code more backends and submit pull requests!

## Fallback call structure

### Forward mode

```mermaid
flowchart LR
    subgraph Gradient
    gradient --> value_and_gradient
    value_and_gradient --> value_and_gradient!
    gradient! --> value_and_gradient!
    end

    subgraph Jacobian
    jacobian --> value_and_jacobian
    value_and_jacobian --> value_and_jacobian!
    jacobian! --> value_and_jacobian!
    end

    subgraph Multiderivative
    multiderivative --> value_and_multiderivative
    value_and_multiderivative --> value_and_multiderivative!
    multiderivative! --> value_and_multiderivative!
    end

    subgraph Derivative
    derivative --> value_and_derivative
    end

    subgraph Pushforward
    pushforward --> value_and_pushforward
    value_and_pushforward --> value_and_pushforward!
    pushforward! --> value_and_pushforward!
    end

    value_and_jacobian! --> value_and_pushforward!
    value_and_gradient! --> value_and_pushforward!
    value_and_multiderivative! --> value_and_pushforward!
    value_and_derivative --> value_and_pushforward
```

### Reverse mode

```mermaid
flowchart LR
    subgraph Gradient
    gradient --> value_and_gradient
    value_and_gradient --> value_and_gradient!
    gradient! --> value_and_gradient!
    end

    subgraph Jacobian
    jacobian --> value_and_jacobian
    value_and_jacobian --> value_and_jacobian!
    jacobian! --> value_and_jacobian!
    end

    subgraph Multiderivative
    multiderivative --> value_and_multiderivative
    value_and_multiderivative --> value_and_multiderivative!
    multiderivative! --> value_and_multiderivative!
    end

    subgraph Derivative
    derivative --> value_and_derivative
    end

    subgraph Pullback
    pullback --> value_and_pullback
    value_and_pullback --> value_and_pullback!
    pullback! --> value_and_pullback!
    end

    value_and_jacobian! --> value_and_pullback!
    value_and_gradient! --> value_and_pullback!
    value_and_multiderivative! --> value_and_pullback!
    value_and_derivative --> value_and_pullback
```

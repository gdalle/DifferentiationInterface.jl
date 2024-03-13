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
    subgraph Pushforward
    pushforward --> value_and_pushforward
    value_and_pushforward --> value_and_pushforward!
    pushforward! --> value_and_pushforward!
    end

    subgraph Derivative
    value_and_derivative --> value_and_pushforward
    derivative --> pushforward
    end
    
    subgraph Multiderivative
    value_and_multiderivative! --> value_and_pushforward!
    value_and_multiderivative --> value_and_pushforward
    multiderivative! --> pushforward!
    multiderivative --> pushforward
    end

    subgraph Gradient
    value_and_gradient! --> pushforward
    value_and_gradient --> value_and_gradient!
    gradient! --> value_and_gradient!
    gradient --> value_and_gradient
    end

    subgraph Jacobian
    value_and_jacobian! --> pushforward!
    value_and_jacobian --> value_and_jacobian!
    jacobian! --> value_and_jacobian!
    jacobian --> value_and_jacobian
    end
```

### Reverse mode

```mermaid
flowchart LR
    subgraph Pullback
    pullback --> value_and_pullback
    value_and_pullback --> value_and_pullback!
    pullback! --> value_and_pullback!
    end
    
    subgraph Derivative
    value_and_derivative --> value_and_pullback
    derivative --> pullback
    end
    
    subgraph Multiderivative
    value_and_multiderivative! --> pullback
    value_and_multiderivative --> value_and_multiderivative!
    multiderivative! --> value_and_multiderivative!
    multiderivative --> value_and_multiderivative
    end

    subgraph Gradient
    value_and_gradient! --> value_and_pullback!
    value_and_gradient --> value_and_pullback
    gradient! --> pullback!
    gradient --> pullback
    end

    subgraph Jacobian
    value_and_jacobian! --> pullback!
    value_and_jacobian --> value_and_jacobian!
    jacobian! --> value_and_jacobian!
    jacobian --> value_and_jacobian
    end
```

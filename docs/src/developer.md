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

!!! note "Edge labels"

    Non-labeled edges in the following graphs correspond to single function calls.

    Edge labels correspond to the amount of function calls when applying operators to a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$.

### Forward mode, allocating functions

```mermaid
flowchart LR
    subgraph Derivative
        value_and_derivative
        derivative
    end

    value_and_derivative --> value_and_pushforward
    derivative --> pushforward
    
    subgraph Multiderivative
        value_and_multiderivative!
        value_and_multiderivative
        multiderivative!
        multiderivative
    end

    value_and_multiderivative! --> value_and_pushforward!
    value_and_multiderivative --> value_and_pushforward
    multiderivative! --> pushforward!
    multiderivative --> pushforward

    subgraph Gradient
        value_and_gradient --> value_and_gradient!
        gradient! --> value_and_gradient!
        gradient --> value_and_gradient
    end

    value_and_gradient! --> |n|pushforward

    subgraph Jacobian
        value_and_jacobian --> value_and_jacobian!
        jacobian! --> value_and_jacobian!
        jacobian --> value_and_jacobian
    end

    value_and_jacobian! --> |n|pushforward!

    subgraph Pushforward
        value_and_pushforward --> value_and_pushforward!
        pushforward! --> value_and_pushforward!
        pushforward --> value_and_pushforward
    end
```

### Forward mode, mutating functions

```mermaid
flowchart LR
    subgraph Multiderivative
        value_and_multiderivative!
    end

    value_and_multiderivative! --> value_and_pushforward!

    subgraph Jacobian
        value_and_jacobian!
    end

    value_and_jacobian! --> |n|value_and_pushforward!

    subgraph Pushforward
        value_and_pushforward!
    end
```

### Reverse mode, allocating functions

```mermaid
flowchart LR
    subgraph Derivative
        value_and_derivative
        derivative
    end

    value_and_derivative --> value_and_pullback
    derivative --> pullback
    
    subgraph Multiderivative
        value_and_multiderivative --> value_and_multiderivative!
        multiderivative! --> value_and_multiderivative!
        multiderivative --> value_and_multiderivative
    end

    value_and_multiderivative! --> |m|pullback

    subgraph Gradient
        value_and_gradient!
        value_and_gradient
        gradient!
        gradient 
    end

    value_and_gradient! --> value_and_pullback!
    value_and_gradient --> value_and_pullback
    gradient! --> pullback!
    gradient --> pullback

    subgraph Jacobian
        value_and_jacobian --> value_and_jacobian!
        jacobian! --> value_and_jacobian!
        jacobian --> value_and_jacobian
    end

    value_and_jacobian! --> |m|pullback!

    subgraph Pullback
        value_and_pullback --> value_and_pullback!
        pullback! --> value_and_pullback!
        pullback --> value_and_pullback
    end
```

### Reverse mode, mutating functions

```mermaid
flowchart LR
    subgraph Multiderivative
        value_and_multiderivative!
    end

    value_and_multiderivative! --> |m|value_and_pullback!

    subgraph Jacobian
        value_and_jacobian!
    end

    value_and_jacobian! --> |m|value_and_pullback!

    subgraph Pullback
        value_and_pullback!
    end
```

### Second order, scalar-valued functions

```mermaid
flowchart LR
    subgraph First order
        gradient!
        value_and_pushforward!
    end

    subgraph Hessian-vector product
        gradient_and_hessian_vector_product!
        gradient_and_hessian_vector_product --> gradient_and_hessian_vector_product!
    end

    gradient_and_hessian_vector_product! --> gradient!
    gradient_and_hessian_vector_product! --> value_and_pushforward!

    subgraph Hessian
        value_and_gradient_and_hessian!
        value_and_gradient_and_hessian --> value_and_gradient_and_hessian!
        hessian! --> value_and_gradient_and_hessian!
        hessian --> value_and_gradient_and_hessian
    end

    value_and_gradient_and_hessian! --> |n|gradient_and_hessian_vector_product!
```

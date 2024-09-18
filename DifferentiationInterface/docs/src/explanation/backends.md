# Backends

## List

We support the following dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

- [`AutoChainRules`](@extref ADTypes.AutoChainRules)
- [`AutoDiffractor`](@extref ADTypes.AutoDiffractor)
- [`AutoEnzyme`](@extref ADTypes.AutoEnzyme)
- [`AutoFastDifferentiation`](@extref ADTypes.AutoFastDifferentiation)
- [`AutoFiniteDiff`](@extref ADTypes.AutoFiniteDiff)
- [`AutoFiniteDifferences`](@extref ADTypes.AutoFiniteDifferences)
- [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff)
- [`AutoGTPSA`](@extref ADTypes.AutoGTPSA)
- [`AutoPolyesterForwardDiff`](@extref ADTypes.AutoPolyesterForwardDiff)
- [`AutoReverseDiff`](@extref ADTypes.AutoReverseDiff)
- [`AutoSymbolics`](@extref ADTypes.AutoSymbolics)
- [`AutoTapir`](@extref ADTypes.AutoTapir)
- [`AutoTracker`](@extref ADTypes.AutoTracker)
- [`AutoZygote`](@extref ADTypes.AutoZygote)

!!! note
    DifferentiationInterface.jl itself is compatible with Julia 1.6, the Long Term Support (LTS) version of the language.
    However, we were only able to test the following backends on Julia 1.6:
    - `AutoFiniteDifferences`
    - `AutoForwardDiff`
    - `AutoReverseDiff`
    - `AutoTracker`
    - `AutoZygote`
    We strongly recommend that users upgrade to Julia 1.10 or above, where all backends are tested.

## Features

Given a backend object, you can use:

- [`check_available`](@ref) to know whether the required AD package is loaded
- [`check_inplace`](@ref) to know whether the backend supports in-place functions (all backends support out-of-place functions)

In theory, all we need from each backend is either a `pushforward` or a `pullback`: we can deduce every other operator from these two.
In practice, many AD backends have custom implementations for high-level operators like `gradient` or `jacobian`, which we reuse whenever possible.

!!! details
    In the rough summary table below,

    - ✅ means that we reuse the custom implementation from the backend;
    - ❌ means that a custom implementation doesn't exist, so we use our default fallbacks;
    - 🔀 means it's complicated or not done yet.

    |                            | `pf` | `pb` | `der` | `grad` | `jac` | `hess` | `hvp` | `der2` |
    | -------------------------- | ---- | ---- | ----- | ------ | ----- | ------ | ----- | ------ |
    | `AutoChainRules`           | ❌    | ✅    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoDiffractor`           | ✅    | ❌    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoEnzyme` (forward)     | ✅    | ❌    | ❌     | ✅      | ✅     | ❌      | ❌     | ❌      |
    | `AutoEnzyme` (reverse)     | ❌    | ✅    | ❌     | ✅      | ✅     | ❌      | 🔀     | ❌      |
    | `AutoFastDifferentiation`  | ✅    | ✅    | ✅     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoFiniteDiff`           | 🔀    | ❌    | ✅     | ✅      | ✅     | ✅      | ❌     | ❌      |
    | `AutoFiniteDifferences`    | 🔀    | ❌    | ❌     | ✅      | ✅     | ❌      | ❌     | ❌      |
    | `AutoForwardDiff`          | ✅    | ❌    | ✅     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoGTPSA`                | ✅    | ❌    | ❌     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoPolyesterForwardDiff` | 🔀    | ❌    | 🔀     | ✅      | ✅     | 🔀      | 🔀     | 🔀      |
    | `AutoReverseDiff`          | ❌    | 🔀    | ❌     | ✅      | ✅     | ✅      | ❌     | ❌      |
    | `AutoSymbolics`            | ✅    | ❌    | ✅     | ✅      | ✅     | ✅      | ❌     | ❌      |
    | `AutoTapir`                | ❌    | ✅    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoTracker`              | ❌    | ✅    | ❌     | ✅      | ❌     | ❌      | ❌     | ❌      |
    | `AutoZygote`               | ❌    | ✅    | ❌     | ✅      | ✅     | ✅      | 🔀     | ❌      |

## Second order

For second-order operators like [`second_derivative`](@ref), [`hessian`](@ref) and [`hvp`](@ref), there are two main options.
You can either use a single backend, or combine two of them within the [`SecondOrder`](@ref) struct:

```julia
backend = SecondOrder(outer_backend, inner_backend)
```

The inner backend will be called first, and the outer backend will differentiate the generated code.
In general, using a forward outer backend over a reverse inner backend will yield the best performance.

!!! danger
    Second-order AD is tricky, and many backend combinations will fail (even if you combine a backend with itself).
    Be ready to experiment and open issues if necessary.

## Backend switch

The wrapper [`DifferentiateWith`](@ref) allows you to switch between backends.
It takes a function `f` and specifies that `f` should be differentiated with the backend of your choice, instead of whatever other backend the code is trying to use.
In other words, when someone tries to differentiate `dw = DifferentiateWith(f, backend1)` with `backend2`, then `backend1` steps in and `backend2` does nothing.
At the moment, `DifferentiateWith` only works when `backend2` supports [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).

## Implementations

What follows is a list of implementation details from the package extensions of DifferentiationInterface.jl
It is not part of the public API or protected by semantic versioning, and it may become outdated.
When in doubt, refer to the code itself.

### ChainRulesCore

We only implement `pullback`, using the [`RuleConfig` mechanism](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html#config) to call back into AD.
Same-point preparation runs the forward sweep and returns the pullback closure.

### Diffractor

We only implement `pushforward`.

!!! danger
    The latest releases of Diffractor [broke DifferentiationInterface](https://github.com/JuliaDiff/Diffractor.jl/issues/290).

### Enzyme

Depending on the `mode` attribute inside [`AutoEnzyme`](@extref ADTypes.AutoEnzyme), we implement either `pushforward` or `pullback` based on `Enzyme.autodiff`.
When necessary, preparation chooses a number of chunks (for `gradient` and `jacobian` in forward mode, for `jacobian` only in reverse mode).

### FastDifferentiation

For every operator, preparation generates an [executable function](https://brianguenter.github.io/FastDifferentiation.jl/stable/makefunction/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

### FiniteDiff

Whenever possible, preparation creates a cache object.

### FiniteDifferences

### ForwardDiff

We implement [`pushforward`](@ref) directly using [`Dual` numbers](https://juliadiff.org/ForwardDiff.jl/stable/dev/how_it_works/), and preparation allocates the necessary space.
For higher level operators, preparation creates a [config object](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers), which can be type-unstable.

### PolyesterForwardDiff

Most operators fall back on `AutoForwardDiff`.

### ReverseDiff

Wherever possible, preparation records a [tape](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API) of the function's execution.

!!! warning
    This tape is specific to the control flow inside the function, and cannot be reused if the control flow is value-dependent (like `if x[1] > 0`).

### Symbolics

For all operators, preparation generates an [executable function](https://docs.sciml.ai/Symbolics/stable/manual/build_function/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

### Tapir

For `pullback`, preparation [builds the reverse rule](https://github.com/withbayes/Tapir.jl?tab=readme-ov-file#how-it-works) of the function.

### Tracker

We implement `pullback` based on `Tracker.back`.
Same-point preparation runs the forward sweep and returns the pullback closure at `x`.

### Zygote

We implement `pullback` based on `Zygote.pullback`.
Same-point preparation runs the forward sweep and returns the pullback closure at `x`.


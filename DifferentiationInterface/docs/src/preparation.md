# Preparation

Preparation is a backend-specific procedure which involves some subtleties.
Here we list the broad principles of preparation for each backend where it is nontrivial.

The following is not part of the public API.

!!! warning
    This page may become outdated, in which case you should refer to the source code as the ground truth.

## ChainRulesCore

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure.

## Enzyme

In forward mode, for [`gradient`](@ref) and [`jacobian`](@ref)

## FastDifferentiation

Preparation generates an [executable function](https://brianguenter.github.io/FastDifferentiation.jl/stable/makefunction/) from the symbolic expression of the differentiated function.

## FiniteDiff

Whenever possible, preparation creates a cache object.

## ForwardDiff

Wherever possible, preparation creates a [config](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers) with all the necessary memory to use as buffer.
For [`pushforward`](@ref), preparation allocates the necessary space for `Dual` number computations.

## ReverseDiff

Wherever possible, preparation records a [tape](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API) of the function's execution.

!!! warning
    This tape is specific to the control flow inside the function, and cannot be reused if the control flow is value-dependent (like `if x[1] > 0`).

## Symbolics

Preparation generates an [executable function](https://docs.sciml.ai/Symbolics/stable/manual/build_function/) from the symbolic expression of the differentiated function.

## Tapir

For [`pullback`](@ref), preparation [builds the reverse rule](https://github.com/withbayes/Tapir.jl?tab=readme-ov-file#how-it-works) of the function.

## Tracker

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure at `x`.

## Zygote

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure at `x`.

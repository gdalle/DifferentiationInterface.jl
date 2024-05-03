#=
Everything in this file is taken from "What color is your Jacobian?"
=#

function get_groups(colors::AbstractVector{<:Integer})
    return map(unique(colors)) do c
        filter(j -> colors[j] == c, eachindex(colors))
    end
end

abstract type AbstractMatrixGraph end

rows(g::AbstractMatrixGraph) = axes(g.A_colmajor, 1)
columns(g::AbstractMatrixGraph) = axes(g.A_colmajor, 2)

## Jacobian coloring

"""
    BipartiteGraph

Represent a bipartite graph between the rows and the columns of a non-symmetric `m × n` matrix `A`.

This graph is defined as `G = (R, C, E)` where `R = 1:m` is the set of row indices, `C = 1:n` is the set of column indices and `(i, j) ∈ E` whenever `A[i, j]` is nonzero.

# Fields

- `A_colmajor::AbstractMatrix`: output of [`col_major`](@ref) applied to `A`
- `A_rowmajor::AbstractMatrix`: output of [`row_major`](@ref) applied to `A`

# Reference

> [What Color Is Your Jacobian? Graph Coloring for Computing Derivatives](https://epubs.siam.org/doi/abs/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct BipartiteGraph{M1<:AbstractMatrix,M2<:AbstractMatrix} <: AbstractMatrixGraph
    A_colmajor::M1
    A_rowmajor::M2

    function BipartiteGraph(A::AbstractMatrix)
        A_colmajor = col_major(A)
        A_rowmajor = row_major(A)
        return new{typeof(A_colmajor),typeof(A_rowmajor)}(A_colmajor, A_rowmajor)
    end
end

neighbors_of_column(g::BipartiteGraph, j::Integer) = nz_in_col(g.A_colmajor, j)
neighbors_of_row(g::BipartiteGraph, i::Integer) = nz_in_row(g.A_rowmajor, i)

function distance2_column_coloring(g::BipartiteGraph)
    n = length(columns(g))
    colors = zeros(Int, n)
    forbidden_colors = zeros(Int, n)
    for v in columns(g)  # default ordering
        for w in neighbors_of_column(g, v)
            for x in neighbors_of_row(g, w)
                if !iszero(colors[x])
                    forbidden_colors[colors[x]] = v
                end
            end
        end
        for c in columns(g)
            if forbidden_colors[c] != v
                colors[v] = c
                break
            end
        end
    end
    return colors
end

function distance2_row_coloring(g::BipartiteGraph)
    m = length(rows(g))
    colors = zeros(Int, m)
    forbidden_colors = zeros(Int, m)
    for v in 1:m  # default ordering
        for w in neighbors_of_row(g, v)
            for x in neighbors_of_column(g, w)
                if !iszero(colors[x])
                    forbidden_colors[colors[x]] = v
                end
            end
        end
        for c in rows(g)
            if forbidden_colors[c] != v
                colors[v] = c
                break
            end
        end
    end
    return colors
end

## Hessian coloring

"""
    AdjacencyGraph

Represent a graph between the columns of a symmetric `n × n` matrix `A` with nonzero diagonal elements.

This graph is defined as `G = (C, E)` where `C = 1:n` is the set of columns and `(i, j) ∈ E` whenever `A[i, j]` is nonzero for some `j ∈ 1:m`, `j ≠ i`.

# Fields

- `A_colmajor::AbstractMatrix`: output of [`col_major`](@ref) applied to `A`

# Reference

> [What Color Is Your Jacobian? Graph Coloring for Computing Derivatives](https://epubs.siam.org/doi/abs/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct AdjacencyGraph{M<:AbstractMatrix} <: AbstractMatrixGraph
    A_colmajor::M

    function AdjacencyGraph(A::AbstractMatrix)
        A_colmajor = col_major(A)
        return new{typeof(A_colmajor)}(A_colmajor)
    end
end

function neighbors(g::AdjacencyGraph, j::Integer)
    return filter(!isequal(j), nz_in_col(g.A_colmajor, j))
end

function colored_neighbors(g::AdjacencyGraph, j::Integer, colors::AbstractVector{<:Integer})
    return filter(neighbors(g, j)) do i
        !iszero(colors[i])
    end
end

function star_coloring(g::AdjacencyGraph)
    n = length(columns(g))
    colors = zeros(Int, n)
    forbidden_colors = zeros(Int, n)
    for v in columns(g)  # default ordering
        for w in neighbors(g, v)
            if !iszero(colors[w])  # w is colored
                forbidden_colors[colors[w]] = v
            end
            for x in neighbors(g, w)
                if !iszero(colors[x]) && iszero(colors[w])  # w is not colored
                    forbidden_colors[colors[x]] = v
                else
                    for y in neighbors(g, x)
                        if !iszero(colors[y]) && y != w
                            if colors[y] == colors[w]
                                forbidden_colors[colors[x]] = v
                                break
                            end
                        end
                    end
                end
            end
        end
        for c in columns(g)
            if forbidden_colors[c] != v
                colors[v] = c
                break
            end
        end
    end
    return colors
end

## ADTypes overloads

"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Matrix coloring algorithm for sparse Jacobians and Hessians.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# See also

- `ADTypes.column_coloring`
- `ADTypes.row_coloring`
- `ADTypes.symmetric_coloring`

# Reference

> [What Color Is Your Jacobian? Graph Coloring for Computing Derivatives](https://epubs.siam.org/doi/abs/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm end

function ADTypes.column_coloring(A, ::GreedyColoringAlgorithm)
    g = BipartiteGraph(A)
    return distance2_column_coloring(g)
end

function ADTypes.row_coloring(A, ::GreedyColoringAlgorithm)
    g = BipartiteGraph(A)
    return distance2_row_coloring(g)
end

function ADTypes.symmetric_coloring(A, ::GreedyColoringAlgorithm)
    g = AdjacencyGraph(A)
    return star_coloring(g)
end

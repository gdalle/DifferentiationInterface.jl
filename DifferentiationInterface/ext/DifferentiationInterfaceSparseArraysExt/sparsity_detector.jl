## Direct

function ADTypes.jacobian_sparsity(f, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    J = jacobian(f, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    J = jacobian(f!, y, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.hessian_sparsity(f, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    H = hessian(f, backend, x)
    return sparse(abs.(H) .> atol)
end

## Iterative

function ADTypes.jacobian_sparsity(f, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    y = f(x)
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if pushforward_performance(backend) isa PushforwardFast
        p = similar(y)
        extras = prepare_pushforward_same_point(
            f, backend, x, Tangents(basis(backend, x, first(eachindex(x))))
        )
        for (kj, j) in enumerate(eachindex(x))
            pushforward!(f, Tangents(p), extras, backend, x, Tangents(basis(backend, x, j)))
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        extras = prepare_pullback_same_point(
            f, backend, x, Tangents(basis(backend, y, first(eachindex(y))))
        )
        for (ki, i) in enumerate(eachindex(y))
            pullback!(f, Tangents(p), extras, backend, x, Tangents(basis(backend, y, i)))
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if pushforward_performance(backend) isa PushforwardFast
        p = similar(y)
        extras = prepare_pushforward_same_point(
            f!, y, backend, x, Tangents(basis(backend, x, first(eachindex(x))))
        )
        for (kj, j) in enumerate(eachindex(x))
            pushforward!(
                f!, y, Tangents(p), extras, backend, x, Tangents(basis(backend, x, j))
            )
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        extras = prepare_pullback_same_point(
            f!, y, backend, x, Tangents(basis(backend, y, first(eachindex(y))))
        )
        for (ki, i) in enumerate(eachindex(y))
            pullback!(
                f!, y, Tangents(p), extras, backend, x, Tangents(basis(backend, y, i))
            )
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.hessian_sparsity(f, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    n = length(x)
    I, J = Int[], Int[]
    p = similar(x)
    extras = prepare_hvp_same_point(
        f, backend, x, Tangents(basis(backend, x, first(eachindex(x))))
    )
    for (kj, j) in enumerate(eachindex(x))
        hvp!(f, Tangents(p), extras, backend, x, Tangents(basis(backend, x, j)))
        for ki in LinearIndices(p)
            if abs(p[ki]) > atol
                push!(I, ki)
                push!(J, kj)
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), n, n)
end

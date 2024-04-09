struct SparseDiffToolsAllocatingJacobianExtras{C} <: JacobianExtras
    cache::C
end

struct SparseDiffToolsHessianExtras{C,E} <: HessianExtras
    inner_gradient_closure::C
    outer_jacobian_extras::E
end

for AutoSparse in SPARSE_BACKENDS
    @eval begin

        ## Jacobian

        function DI.prepare_jacobian(f, backend::$AutoSparse, x::AbstractArray)
            sparsity_detector::AbstractMaybeSparsityDetection =
                if hasfield(typeof(backend), :sparsity_detector) &&
                    !isnothing(backend.sparsity_detector)
                    backend.sparsity_detector
                else
                    SymbolicsSparsityDetection()
                end
            cache = sparse_jacobian_cache(backend, sparsity_detector, f, x; fx=f(x))
            return SparseDiffToolsAllocatingJacobianExtras(cache)
        end

        function DI.value_and_jacobian!!(
            f, jac, backend::$AutoSparse, x, extras::SparseDiffToolsAllocatingJacobianExtras
        )
            sparse_jacobian!(jac, backend, extras.cache, f, x)
            return f(x), jac
        end

        function DI.jacobian!!(
            f, jac, backend::$AutoSparse, x, extras::SparseDiffToolsAllocatingJacobianExtras
        )
            sparse_jacobian!(jac, backend, extras.cache, f, x)
            return jac
        end

        function DI.value_and_jacobian(
            f, backend::$AutoSparse, x, extras::SparseDiffToolsAllocatingJacobianExtras
        )
            return f(x), sparse_jacobian(backend, extras.cache, f, x)
        end

        function DI.jacobian(
            f, backend::$AutoSparse, x, extras::SparseDiffToolsAllocatingJacobianExtras
        )
            return sparse_jacobian(backend, extras.cache, f, x)
        end

        ## Hessian

        function DI.prepare_hessian(f, backend::SecondOrder{<:$AutoSparse}, x)
            inner_gradient_closure(z) = DI.gradient(f, inner(backend), z)
            outer_jacobian_extras = DI.prepare_jacobian(
                inner_gradient_closure, outer(backend), x
            )
            return SparseDiffToolsHessianExtras(
                inner_gradient_closure, outer_jacobian_extras
            )
        end

        function DI.hessian(
            f, backend::SecondOrder{<:$AutoSparse}, x, extras::SparseDiffToolsHessianExtras
        )
            return DI.jacobian(
                extras.inner_gradient_closure,
                outer(backend),
                x,
                extras.outer_jacobian_extras,
            )
        end

        function DI.hessian!!(
            f,
            hess,
            backend::SecondOrder{<:$AutoSparse},
            x,
            extras::SparseDiffToolsHessianExtras,
        )
            return DI.jacobian!!(
                extras.inner_gradient_closure,
                hess,
                outer(backend),
                x,
                extras.outer_jacobian_extras,
            )
        end
    end
end

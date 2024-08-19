DI.check_available(backend::AutoSparse) = DI.check_available(dense_ad(backend))
DI.twoarg_support(backend::AutoSparse) = DI.twoarg_support(dense_ad(backend))

function DI.pushforward_performance(backend::AutoSparse)
    return DI.pushforward_performance(dense_ad(backend))
end

DI.pullback_performance(backend::AutoSparse) = DI.pullback_performance(dense_ad(backend))
DI.hvp_mode(backend::AutoSparse{<:SecondOrder}) = DI.hvp_mode(dense_ad(backend))

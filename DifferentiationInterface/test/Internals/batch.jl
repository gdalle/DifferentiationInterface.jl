import DifferentiationInterface as DI
using Test

@test DI.pick_batchsize.(1:(DI.DEFAULT_BATCHSIZE)) == 1:(DI.DEFAULT_BATCHSIZE)
@test all(
    DI.pick_batchsize.((DI.DEFAULT_BATCHSIZE + 1):(5DI.DEFAULT_BATCHSIZE)) .<=
    DI.DEFAULT_BATCHSIZE,
)
@test all(
    DI.pick_batchsize.((DI.DEFAULT_BATCHSIZE + 1):(5DI.DEFAULT_BATCHSIZE)) .>=
    DI.DEFAULT_BATCHSIZE / 2,
)

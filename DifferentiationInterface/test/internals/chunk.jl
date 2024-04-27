@test DI.pick_chunksize.(1:(DI.DEFAULT_CHUNKSIZE)) == 1:(DI.DEFAULT_CHUNKSIZE)
@test all(
    DI.pick_chunksize.((DI.DEFAULT_CHUNKSIZE + 1):(5DI.DEFAULT_CHUNKSIZE)) .<=
    DI.DEFAULT_CHUNKSIZE,
)
@test all(
    DI.pick_chunksize.((DI.DEFAULT_CHUNKSIZE + 1):(5DI.DEFAULT_CHUNKSIZE)) .>=
    DI.DEFAULT_CHUNKSIZE / 2,
)


chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")

if (nzchar(chk) && chk == "TRUE") {
  # use 2 cores in CHECK/Travis/AppVeyor
  num_workers <- 2L
} else {
  # use all cores in devtools::test()
  num_workers <- parallel::detectCores()
}

test_that("Parallelization works as expected", {
  skip_on_cran()
  skip_on_bioc()
  skip_on_covr()
  skip_on_ci()

  set.seed(1)
  n = 100
  x = rnorm(n = n)
  df = data.frame(
    x = x,
    y1 = as.integer(seq_along(x)),
    y2 = sample(c('test', 'tset'), size = n, replace = TRUE)
  )

  result_predictors = score_predictors(df, 'y1', do_parallel = TRUE, nc = num_workers)
  result_df = score_df(df, do_parallel = TRUE, nc = num_workers)
  expect_true(is.data.frame(result_predictors))
  expect_equal(nrow(result_predictors), ncol(df))
  expect_true(is.data.frame(result_df))
  expect_equal(nrow(result_df), ncol(df) ^ 2)
})

context("Check whether PPS calculations are correct")

test_that("PPS is between zero and one", {
  set.seed(1)
  n = 100
  x = rnorm(n = n)
  df = data.frame(
    x = x,
    y1 = rnorm(n = n),
    y2 = runif(n = n),
    y3 = x,
    y4 = x ^ 2 + rnorm(n = n),
    y5 = as.numeric(x > 0)
  )

  pps1 = score(df, 'x', 'y1', verbose = FALSE)[['pps']]
  pps2 = score(df, 'x', 'y2', verbose = FALSE)[['pps']]
  pps3 = score(df, 'x', 'y3', verbose = FALSE)[['pps']]
  pps4 = score(df, 'x', 'y4', verbose = FALSE)[['pps']]
  pps5 = score(df, 'x', 'y5', verbose = FALSE)[['pps']]

  min_pps = 0
  max_pps = 1

  expect_true(pps1 >= min_pps & pps1 <= max_pps)
  expect_true(pps2 >= min_pps & pps2 <= max_pps)
  expect_true(pps3 >= min_pps & pps3 <= max_pps)
  expect_true(pps4 >= min_pps & pps4 <= max_pps)
  expect_true(pps5 >= min_pps & pps4 <= max_pps)
})


test_that("The calculated PPS is stable", {
  set.seed(1)
  n = 100
  x = rnorm(n = n)
  df = data.frame(
    x = x,
    y1 = x ^ 2 + rnorm(n = n),
    y2 = as.numeric(x > 0)
  )

  pps1.1 = score(df, 'x', 'y1', verbose = FALSE)[['pps']]
  pps2.1 = score(df, 'x', 'y2', verbose = FALSE)[['pps']]

  pps1.2 = score(df, 'x', 'y1', verbose = FALSE)[['pps']]
  pps2.2 = score(df, 'x', 'y2', verbose = FALSE)[['pps']]

  expect_equal(pps1.1, pps1.2)
  expect_equal(pps2.1, pps2.2)
})

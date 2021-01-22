context("Check whether column identification works")

test_that("ID columns are identified", {

  df = mtcars
  df[['id']] = row.names(mtcars)

  expect_true(is_id(df[['id']]))
  expect_equal(score(df, x = 'id', y = 'mpg')[['pps']], 0)
  expect_equal(score(df, x = 'mpg', y = 'id')[['pps']], 0)
})

test_that("Constants are identified", {

  df = mtcars
  df$constant1 = 1
  df$constant2 = 'a'

  expect_true(is_constant(df$constant1))
  expect_true(is_constant(df$constant2))
  expect_false(is_constant(df$mpg))
  expect_equal(score(df, x = 'constant1', y = 'constant2')[['pps']], 0)
  expect_equal(score(df, x = 'constant2', y = 'constant1')[['pps']], 0)
})

test_that("Identify when target and features are the same", {

  df = mtcars
  df$mpg2 = df$mpg

  expect_true(is_same(df$mpg, df$mpg2))
  expect_false(is_same(df$mpg, df$cyl))
  expect_equal(score(df, x = 'mpg', y = 'mpg')[['pps']], 1)
  expect_equal(score(df, x = 'mpg', y = 'mpg2')[['pps']], 1)
})

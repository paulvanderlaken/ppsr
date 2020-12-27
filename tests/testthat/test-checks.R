context("Check column characteristics")

test_that("ID columns are identified", {
  expect_true(is_id(row.names(mtcars)))
  expect_equal(score(x = row.names(mtcars), y = mtcars$mpg), 0)
  expect_equal(score(x = mtcars$mpg, y = row.names(mtcars)), 0)
})

test_that("Constants are identified", {

  column_constant1 = c(1, 1, 1)
  column_constant2 = c('a', 'a', 'a')
  column_variance1 = c(1, 1, 2)
  column_variance2 = c('a', 'a', 'b')

  expect_true(is_constant(column_constant1))
  expect_true(is_constant(column_constant2))
  expect_false(is_constant(column_variance1))
  expect_equal(score(x = column_constant1, y = column_variance1), 0)
  expect_equal(score(x = column_variance1, y = column_constant1), 0)
})

test_that("Identify when target and features are the same", {

  column1 = c(1, 2, 3)
  column2 = c('a', 'a', 'b')

  expect_true(is_same(column1, column1))
  expect_true(is_same(column2, column2))
  expect_false(is_same(column1, column2))
  expect_equal(score(x = column1, y = column1), 1)
  expect_equal(score(x = column2, y = column2), 1)
})

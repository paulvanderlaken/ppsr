context("Check whether visualizations work")

plot_pred = visualize_predictors(iris, 'Species')
plot_mat = visualize_matrix(iris)
plot_cor = visualize_correlations(iris)
plot_both = visualize_both(iris)

test_that("Functions run without error", {
  iris_subset = iris[, 1:2]
  expect_success(visualize_predictors(iris, 'Species'))
  expect_success(visualize_matrix(iris_subset))
  expect_success(visualize_correlations(iris))
  expect_success(visualize_both(iris_subset))
})



test_that("Functions produce ggplot2 lists", {
  expect_true(is.list(plot_pred))
  expect_true(is.list(plot_mat))
  expect_true(is.list(plot_cor))
  expect_true(is.list(plot_both))
})


test_that("Functions produce ggplot2 plots and return nothing else", {
  expect_true(expect_visible(plot_pred))
  expect_true(expect_visible(plot_mat))
  expect_true(expect_visible(plot_cor))
  expect_true(expect_visible(plot_both))
})

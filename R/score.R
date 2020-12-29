#' Calculate predictive power score for x on y
#'
#' @param x any type of vector of predictor values
#' @param y any type of vector of target values
#' @param algorithm character scalar
#' @param eval_funs named list of \code{eval_*} functions used for
#'     regression and classification problems
#'
#' @return numeric scalar representing predictive power score
#' @export
#'
#' @examples
#' score(x = mtcars$mpg, y = mtcars$cyl)
#'
#' score(x = iris$Petal.Length, y = iris$Species)
score = function(x,
                 y,
                 algorithm = 'tree',
                 eval_funs = list('regression' = eval_mae, 'classification' = eval_f1_score)) {
  # perform checks
  if (is_id(x) | is_id(y)) {
    return(0) # an ID variable has no predictive power
  }
  if (is_constant(x) | is_constant(y)) {
    return(0) # a vector without variation has no predictive power
  }
  if (is_same(x, y)) {
    return(1) # a vector that is completely similar has full predictive power
  }

  # store predictor and target in data frame
  df = data.frame(x = x, y = y)

  # drop any rows where target or predictor is missing
  df = df[stats::complete.cases(df), ]

  # determine type of model we are dealing with
  mode = ifelse(is.numeric(y) & !is_binary(y), 'regression', 'classification')

  # specify and set up statistical model
  # TODO implement other models

  if (algorithm == 'tree') {
    model = parsnip::decision_tree()
    model = parsnip::set_engine(model, "rpart")
  }

  # finalize model setup, fit the model, and predict values
  model = parsnip::set_mode(object = model, mode = mode)
  model = parsnip::fit(model, formula = y ~ x, data = df)
  yhat = stats::predict(model, new_data = df)[[1]]
  # TODO implement cross-validation

  # get the appropriate evaluation function
  eval_fun = eval_funs[[mode]]

  # calculate prediction error
  score_original = eval_fun(y = y, yhat = yhat)

  # calculate prediction error of a naive model
  if (mode == 'regression') {
    # naive regression model takes the mean value of the target variable
    naive_predictions = mean(df$y)
    score_naive = eval_fun(y = df$y, yhat = naive_predictions)

    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    score_normalized = max(c(1 - (score_original / score_naive), 0))
  } else {
    # naive classification model takes the best model
    # among a model that predicts random values
    # and a model that predicts the most common case
    naive_predictions_model = rep(modal_value(df$y), times = nrow(df))
    naive_predictions_random = sample(df$y, size = nrow(df), replace = TRUE)
    score_naive = max(c(eval_fun(y = df$y, yhat = naive_predictions_model),
                        eval_fun(y = df$y, yhat = naive_predictions_random)))

    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    score_normalized = max(c((score_original - score_naive) / (1 - score_naive), 0))
  }

  return(score_normalized)
}

#' Calculate predictive power scores for y
#' Calcuates the predictive power scores for the specified \code{y} variable
#' using every column in the dataset as \code{x}, including itself.
#'
#' @param df data.frame
#' @param y character scalar with target column name
#' @param ... any arguments passed to \code{\link{score}}
#'
#' @return list of float scalars, representing predictive power scores
#' @export
#'
#' @examples
#' score_predictors(df = mtcars, y = 'mpg')
#'
#' score_predictors(df = iris, y = 'Species')
score_predictors = function(df, y, ...) {
  scores = list()
  cnames = colnames(df)
  for (x in cnames) {
    scores[x] = score(x = df[[x]], y = df[[y]], ...)
  }
  return(scores)
}

#' Calculate predictive power score matrix
#' Iterates through the columns of the dataset, calculating the predictive power
#' score for every possible combination of \code{x} and \code{y}.
#'
#' Note that the targets are on the rows, and the features on the columns.
#'
#' @param df data.frame
#' @param ... any arguments passed to \code{\link{score}}
#'
#' @return matrix of floats, representing predictive power scores
#' @export
#'
#' @examples
#' score_matrix(mtcars)
#'
#' score_matrix(iris)
score_matrix = function(df, ...) {
  cnames = colnames(df)
  mtrx = matrix(nrow = ncol(df), ncol = ncol(df), dimnames = list(cnames, cnames))
  for (target in cnames) {
    for (predictor in cnames) {
      mtrx[target, predictor] = score(x = df[[predictor]], y = df[[target]], ...)
    }
  }
  return(mtrx)
}

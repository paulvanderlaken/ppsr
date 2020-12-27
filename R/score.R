#' Calculate predictive power score for x on y
#'
#' @param x any type of vector of predictor values
#' @param y any type of vector of target values
#' @param algorithm character scalar
#' @param metric model evaluation function that takes arguments \code{y} and \code{yhat} and returns a numeric scalar
#'
#' @return numeric scalar representing predictive power score
#' @export
#'
#' @examples
#' score(x = mtcars$mpg, y = mtcars$cyl, algorithm = 'tree', metric = mae)
score = function(x = mtcars$mpg, y = mtcars$cyl, algorithm = 'tree', metric = mae) {
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

  # specify and set up statistical model
  if (algorithm == 'tree') {
    model = parsnip::decision_tree() %>%
      parsnip::set_engine("rpart")
  }
  # TODO implement other models

  mode = ifelse(is.numeric(y), 'regression', 'classification')
  model = parsnip::set_mode(object = model, mode = mode)

  # fit the model and predict the values
  model = parsnip::fit(model, formula = y ~ x, data = df)
  yhat = stats::predict(model, new_data = df)[[1]]
  # TODO implement cross-validation

  # evaluate predictive performance performance
  score_original = metric(y = y, yhat = yhat)
  score_naive = metric(y = y, yhat = mean(y))
  score_normalized = 1 - score_original / score_naive

  return(score_normalized)
}

#' Calculate predictive power scores for y
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
score_predictors = function(df, y, ...) {
  scores = list()
  cnames = colnames(df)
  for (x in cnames) {
    scores[x] = score(x = df[[x]], y = df[[y]], ...)
  }
  return(scores)
}

#' Calculate predictive power score matrix
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

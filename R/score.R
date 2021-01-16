
#' Calculates out-of-sample model performance of a statistical model
#'
#' @param train df, training data, containing variable y
#' @param test df, test data, containing variable y
#' @param model parsnip model object, with mode preset
#' @param eval_fun function, evaluation function being used
#'
#' @return float, evaluation score for predictions using naive model
score_model = function(train, test, model, eval_fun) {
  model = parsnip::fit(model, formula = y ~ x, data = train)
  yhat = stats::predict(model, new_data = test)[[1]]
  return(eval_fun(y = test$y, yhat = yhat))
}

#' Calculate out-of-sample model performance of naive baseline model
#' The calculation that's being performed depends on the type of model
#' For regression models, the mean is used as prediction
#' For classification, a model predicting random values and
#' a model predicting modal values are used and
#' the best model is taken as baseline score
#'
#' @param train df, training data, containing variable y
#' @param test df, test data, containing variable y
#' @param type character, type of model
#' @param eval_fun function, evaluation function being used
#'
#' @return float, evaluation score for predictions using naive model
score_naive = function(train, test, type, eval_fun) {
  if (type == 'regression') {
    # naive regression model takes the mean value of the target variable
    naive_predictions = rep(mean(train$y), nrow(test))
    return(eval_fun(y = test$y, yhat = naive_predictions))
  } else {
    # naive classification model takes the best model
    # among a model that predicts the most common case
    # and a model that predicts random values
    naive_predictions_modal = rep(modal_value(train$y), times = nrow(test))
    # ensure that the random predictions are always the same
    naive_predictions_random = sample(train$y, size = nrow(test), replace = TRUE)
    return(max(c(eval_fun(y = test$y, yhat = naive_predictions_modal),
                 eval_fun(y = test$y, yhat = naive_predictions_random))))
  }
}

#' Normalizes the original score compared to a naive baseline score
#' The calculation that's being performed depends on the type of model
#'
#' @param score_baseline float, the evaluation metric score for a naive model
#' @param score_original float, the evaluation metric score for a statistical model
#' @param type character, type of model
#'
#' @return float, normalized score
normalize_score = function(score_baseline, score_original, type) {
  if (type == 'regression') {
    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    return(max(c(1 - (score_original / score_baseline), 0)))
  } else {
    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    return(max(c((score_original - score_baseline) / (1 - score_baseline), 0)))
  }
}

#' Calculate predictive power score for x on y
#'
#' @param x any type of vector of predictor values
#' @param y any type of vector of target values
#' @param algorithm character scalar
#' @param eval_funs named list of \code{eval_*} functions used for
#'     regression and classification problems
#' @param cv_folds float, number of cross-validation folds
#' @param seed float, seed to ensure reproducibility/stability
#'
#' @return float,  representing predictive power score
#' @export
#'
#' @examples
#' score(x = mtcars$mpg, y = mtcars$cyl)
#'
#' score(x = iris$Petal.Length, y = iris$Species)
score = function(x,
                 y,
                 algorithm = 'tree',
                 eval_funs = list('regression' = eval_mae, 'classification' = eval_weighted_f1_score),
                 cv_folds = 5,
                 seed = 1) {

  ## remove all missings
  is_any_missing = is.na(x) | is.na(y)
  is_complete = !is_any_missing
  x = x[is_complete]
  y = y[is_complete]

  ## perform checks
  # if there's no data left, there's no predictive power
  if (sum(is_complete) == 0) {
    return(0)
  }

  # an ID variable has no predictive power
  if (is_id(x) | is_id(y)) {
    return(0)
  }
  # a vector without variation has no predictive power
  if (is_constant(x) | is_constant(y)) {
    return(0)
  }
  # a vector that is completely similar has full predictive power
  if (is_same(x, y)) {
    return(1)
  }

  # force binary numerics into factors
  if (is_binary_numeric(y)) {
    warning('Forcing binary numeric to factor.\n')
    y = as.factor(y)
  }

  # check whether the number of cv_folds is possible
  if (cv_folds < 1) {
    stop('cv_folds needs to be numeric and larger or equal to 1')
  }
  if (cv_folds > length(y)) {
    stop('There are more cv_folds than unique x/y values. Pick a smaller number of folds.')
  }
  n_per_fold = length(y) / cv_folds
  if (n_per_fold < 10) {
    warning('There are on average only ', n_per_fold, ' observations in each test-set.\n',
    'Model performance will be highly instable. Fewer cv_folds is advised.')
  }

  # set seed to ensure stability of results
  set.seed(seed)

  ## set up statistical model
  # TODO implement other models
  if (algorithm == 'tree') {
    model = parsnip::decision_tree()
    model = parsnip::set_engine(model, "rpart")
  }
  # determine type of model we are dealing with
  type = ifelse(is.numeric(y), 'regression', 'classification')
  model = parsnip::set_mode(object = model, mode = type)
  # get the appropriate evaluation function
  eval_fun = eval_funs[[type]]


  ## prepare data
  # store predictor and target in data frame
  df = data.frame(x = x, y = y)
  # make cross validation folds
  if (cv_folds > 1) {
    membership = 1 + seq_len(nrow(df)) %% cv_folds # create vector with equal amount of group ids
    random_membership = sample(membership) # shuffle the group ids
    # split the training and test sets
    folds = lapply(seq_len(cv_folds), function(test_fold){
      inTest = random_membership == test_fold
      return(list(
        fold = test_fold,
        test = df[inTest, ],
        train = df[!inTest, ])
      )
    })
  } else if (cv_folds == 1) {
    folds = list(
      fold = 1,
      test = df,
      train = df)
  }


  # evaluate model for each cross validation
  # in the Python implementation there is only one baseline calculated
  # yet I feel that it would be better to assess each folds predictive performance
  # on a baseline that also resembles the naive performance on that test set
  scores_normalized = vapply(folds, FUN = function(x) {
    score_original = score_model(train = x[['train']],
                                 test = x[['test']],
                                 model = model,
                                 eval_fun = eval_fun)
    score_baseline = score_naive(train = x[['train']],
                                 test = x[['test']],
                                 type = type,
                                 eval_fun = eval_fun)
    return(normalize_score(score_baseline = score_baseline,
                           score_original = score_original,
                           type = type))
  }, FUN.VALUE = as.numeric(1))

  return(mean(scores_normalized))
}

#' Calculate predictive power scores for y
#' Calculates the predictive power scores for the specified \code{y} variable
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

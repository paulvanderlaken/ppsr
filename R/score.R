
#' Calculate predictive power score for x on y
#'
#' @param df data.frame containing columns for x and y
#' @param x character, column name of predictor variable
#' @param y character, column name of target variable
#' @param algorithm character
#' @param metrics named list of \code{eval_*} functions used for
#'     regression and classification problems
#' @param cv_folds float, number of cross-validation folds
#' @param seed float, seed to ensure reproducibility/stability
#' @param verbose bool, whether to print notifications
#'
#' @return float,  representing predictive power score
#' @export
#'
#' @examples
#' score(iris, x = 'Petal.Length', y = 'Species')
score = function(df,
                 x,
                 y,
                 algorithm = 'tree',
                 metrics = list('regression' = 'MAE', 'classification' = 'F1_weighted'),
                 cv_folds = 5,
                 seed = 1,
                 verbose = TRUE) {

  # check if x and y are different variables
  if (x == y) {
    return(generate_invalid_report(x, y, 'predictor and target are the same', 1))
  }

  # check if columns occur in dataframe
  cnames = colnames(df)
  for (name in c(x, y)) {
    if (! name %in% cnames) {
      stop(name, ' is not a column in the provided data frame')
    } else if (sum(cnames == name) > 1) {
      stop(name, ' appears as a column name more than once in the data frame')
    }
  }

  # drop all other columns from dataframe
  df = df[, c(x, y)]
  # remove records with missing data
  df = df[stats::complete.cases(df), ]

  # if there's no data left, there's no predictive power
  if (nrow(df) == 0) {
    return(generate_invalid_report(x, y, 'no non-missing records', 1))
  }


  # an ID variable has no predictive power
  if (is_id(df[[y]])) {
    return(generate_invalid_report(x, y, 'target is id', 0))
  } else if (is_id(df[[x]])) {
    return(generate_invalid_report(x, y, 'predictor is id', 0))
  }
  # a vector without variation has no predictive power
  if (is_constant(df[[y]])) {
    return(generate_invalid_report(x, y, 'target is constant', 0))
  } else if (is_constant(df[[x]])) {
    return(generate_invalid_report(x, y, 'predictor is constant', 0))
  }
  # a vector that is completely similar has full predictive power
  if (is_same(df[[x]], df[[y]])) {
    return(generate_invalid_report(x, y, 'target and predictor are same', 1))
  }

  # force binary numerics into factors
  if (is_binary_numeric(df[[y]])) {
    if(verbose) cat('Note: ', y, 'was forced from binary numeric to factor.\n')
    df[[y]] = as.factor(df[[y]])
  }

  # check whether the number of cvfolds is possible and sensible
  if (cv_folds < 1) {
    stop('cv_folds needs to be numeric and larger or equal to 1')
  }
  if (cv_folds > length(df[[y]])) {
    stop('There are more cv_folds than unique ', x, '-', y, ' values. Pick a smaller number of folds.')
  }
  n_per_fold = length(df[[y]]) / cv_folds
  if (n_per_fold < 10) {
    warning('There are on average only ', n_per_fold, ' observations in each test-set',
            ' for the ', x, '-', y, ' relationship.\n',
            'Model performance will be highly instable. Fewer cv_folds are advised.')
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
  type = ifelse(is.numeric(df[[y]]), 'regression', 'classification')
  model = parsnip::set_mode(object = model, mode = type)
  # get the appropriate evaluation function
  # check if metrics are provided
  if (! 'regression' %in% names(metrics)) {
    stop('Input list of metrics does not contain a metric for regression')
  }
  if (! 'classification' %in% names(metrics)) {
    stop('Input list of metrics does not contain a metric for classification')
  }
  metric = metrics[[type]]

  ## prepare data
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

  ## evaluate model in each cross validation
  ## TODO simplify this into a get_scores function
  # in the Python implementation there is only one baseline calculated
  # yet I feel that it would be better to assess each folds predictive performance
  # on a baseline that also resembles the naive performance on that same test set
  scores = lapply(folds, FUN = function(e) {
    model_score = score_model(train = e[['train']],
                              test = e[['test']],
                              model = model,
                              x,
                              y,
                              metric = metric)
    baseline_score = score_naive(train = e[['train']],
                                 test = e[['test']],
                                 x,
                                 y,
                                 type = type,
                                 metric = metric)
    normalized_score = normalize_score(baseline_score = baseline_score,
                                       model_score = model_score,
                                       type = type)
    return(list(
      model_score = model_score,
      baseline_score = baseline_score,
      normalized_score = normalized_score
    ))
  })

  report = list(
    x = x,
    y = y,
    result_type = 'predictive power score',
    pps = mean(vapply(scores, function(x) x$normalized_score, numeric(1))),
    metric = metric,
    baseline_score = mean(vapply(scores, function(x) x$baseline_score, numeric(1))),
    model_score = mean(vapply(scores, function(x) x$model_score, numeric(1))),
    cv_folds = cv_folds,
    seed = seed,
    algorithm = algorithm,
    #TODO: Find out how to store model in the resulting report
    #model = model, #Error: All columns in a tibble must be vectors.
    model_type = type
  )

  return(report)
}

#' Calculate predictive power scores for y
#' Calculates the predictive power scores for the specified \code{y} variable
#' using every column in the dataset as \code{x}, including itself.
#'
#' @inheritParams score
#' @param ... any arguments passed to \code{\link{score}}
#'
#' @return dataframe, detailed report on predictive power scores
#' @export
#'
#' @examples
#' score_predictors(df = mtcars, y = 'mpg')
#'
#' score_predictors(df = iris, y = 'Species')
score_predictors = function(df, y, ...) {
  #TODO implement parallelization here
  scores = lapply(colnames(df), function(x) {
    score(df, x = x, y = y, ...)
  })
  scores = fill_blanks_in_list(scores)
  scores_df = do.call(rbind.data.frame, scores)
  rownames(scores_df) = NULL #TODO fix issue with duplicate rownames
  return(scores_df)
}

#' Calculate predictive power scores for whole dataframe
#' Iterates through the columns of the dataframe, calculating the predictive power
#' score for every possible combination of \code{x} and \code{y}.
#'
#' @inheritParams score
#' @param ... any arguments passed to \code{\link{score}}
#'
#' @return dataframe, detailed report on predictive power scores
#' @export
#'
#' @examples
#' score_df(iris)
score_df = function(df, ...) {
  cnames = colnames(df)
  g = expand.grid(x = cnames, y = cnames, stringsAsFactors = FALSE)
  #TODO implement parallelization here or make sure to leverage it at lower levels
  scores = lapply(seq_len(nrow(g)), function(i) {
    score(df, x = g[['x']][i], y = g[['y']][i], ...)
  })
  scores = fill_blanks_in_list(scores)
  df_scores = do.call(rbind.data.frame, scores)
  rownames(df_scores) = NULL #TODO fix issue with duplicate rownames
  return(df_scores)
}


#' Calculate predictive power score matrix
#' Iterates through the columns of the dataset, calculating the predictive power
#' score for every possible combination of \code{x} and \code{y}.
#'
#' Note that the targets are on the rows, and the features on the columns.
#'
#' @inheritParams score
#' @param ... any arguments passed to \code{\link{score}}
#'
#' @return matrix of floats, representing predictive power scores
#' @export
#'
#' @examples
#' score_matrix(iris)
score_matrix = function(df, ...) {
  df_scores = score_df(df, ...)
  var_uq = unique(df_scores[['x']])
  mtrx = matrix(nrow = length(var_uq), ncol = length(var_uq), dimnames = list(var_uq, var_uq))
  for (x in var_uq) {
    for (y in var_uq) {
      # Note: target on the y axis (rows) and feature on the x axis (columns)
      mtrx[y, x] = df_scores[['pps']][df_scores[['x']] == x & df_scores[['y']] == y]
    }
  }
  return(mtrx)
}



#' Calculate correlation coefficients for whole dataframe
#'
#' @param df data.frame containing columns for x and y
#' @param ... arguments to pass to \code{\link{stats::cor}}
#'
#' @return data.frame with x-y correlation coefficients
#' @export
#'
#' @examples
#' score_correlations(iris)
score_correlations = function(df, ...) {
  isCorrelationColumn = vapply(df, function(x) is.numeric(x) | is.logical(x), logical(1))
  cnames = names(df)[isCorrelationColumn]
  cmat = stats::cor(df[, cnames], ...)
  correlation = as.vector(cmat)
  y = rep(rownames(cmat), each = ncol(cmat))
  x = rep(colnames(cmat), times = nrow(cmat))
  return(data.frame(x, y, correlation))
}





generate_invalid_report = function(x, y, result_type, pps) {
  return(list(
    x = x,
    y = y,
    result_type = result_type,
    pps = pps
  ))
}

#' Calculates out-of-sample model performance of a statistical model
#'
#' @param train df, training data, containing variable y
#' @param test df, test data, containing variable y
#' @param model parsnip model object, with mode preset
#' @param x character, column name of predictor variable
#' @param y character, column name of target variable
#' @param metric character, name of evaluation metric being used, see \code{evaluation_functions()}
#'
#' @return float, evaluation score for predictions using naive model
score_model = function(train, test, model, x, y, metric) {
  model = parsnip::fit(model,
                       formula = stats::as.formula(paste(y, '~', x)),
                       data = train)
  yhat = stats::predict(model, new_data = test)[[1]]
  eval_fun = evaluation_functions()[[metric]]
  return(eval_fun(y = test[[y]], yhat = yhat))
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
#' @param x character, column name of predictor variable
#' @param y character, column name of target variable
#' @param type character, type of model
#' @param metric character, evaluation metric being used
#'
#' @return float, evaluation score for predictions using naive model
score_naive = function(train, test, x, y, type, metric) {
  eval_fun = evaluation_functions()[[metric]]
  if (type == 'regression') {
    # naive regression model takes the mean value of the target variable
    naive_predictions = rep(mean(train[[y]]), nrow(test))
    return(eval_fun(y = test[[y]], yhat = naive_predictions))
  } else {
    # naive classification model takes the best model
    # among a model that predicts the most common case
    # and a model that predicts random values
    naive_predictions_modal = rep(modal_value(train[[y]]), times = nrow(test))
    # ensure that the random predictions are always the same
    naive_predictions_random = sample(train[[y]], size = nrow(test), replace = TRUE)
    return(max(c(eval_fun(y = test[[y]], yhat = naive_predictions_modal),
                 eval_fun(y = test[[y]], yhat = naive_predictions_random))))
  }
}

#' Normalizes the original score compared to a naive baseline score
#' The calculation that's being performed depends on the type of model
#'
#' @param baseline_score float, the evaluation metric score for a naive baseline (model)
#' @param model_score float, the evaluation metric score for a statistical model
#' @param type character, type of model
#'
#' @return float, normalized score
normalize_score = function(baseline_score, model_score, type) {
  if (type == 'regression') {
    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    return(max(c(1 - (model_score / baseline_score), 0)))
  } else {
    # normalize the pps by taking the relative improvement over a naive model
    # or 0 in case of worse performance than a naive model
    return(max(c((model_score - baseline_score) / (1 - baseline_score), 0)))
  }
}


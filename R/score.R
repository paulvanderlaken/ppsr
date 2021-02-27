
#' Calculate predictive power score for x on y
#'
#' @param df data.frame containing columns for x and y
#' @param x string, column name of predictor variable
#' @param y string, column name of target variable
#' @param algorithm string, see \code{available_algorithms()}
#' @param metrics named list of \code{eval_*} functions used for
#'     regression and classification problems, see \code{available_evaluation_metrics()}
#' @param cv_folds float, number of cross-validation folds
#' @param seed float, seed to ensure reproducibility/stability
#' @param verbose boolean, whether to print notifications
#'
#' @return a named list, potentially containing \describe{
#'     \item{x}{the name of the predictor variable}
#'     \item{y}{the name of the target variable}
#'     \item{result_type}{text showing how to interpret the resulting score}
#'     \item{pps}{the predictive power score}
#'     \item{metric}{the evaluation metric used to compute the PPS}
#'     \item{baseline_score}{the score of a naive model on the evaluation metric}
#'     \item{model_score}{the score of the predictive model on the evaluation metric}
#'     \item{cv_folds}{how many cross-validation folds were used}
#'     \item{seed}{the seed that was set}
#'     \item{algorithm}{text shwoing what algorithm was used}
#'     \item{model_type}{text showing whether classification or regression was used}
#' }
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

  # force binary numerics, boolean/logicals, and characters/texts to factor
  if (is_binary_numeric(df[[y]]) | is.logical(df[[y]]) | is.character(df[[y]])) {
    if(verbose) cat('Note:', y, 'was forced from', typeof(df[[y]]), 'to factor.\n')
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
  # determine type of model we are dealing with
  type = ifelse(is.numeric(df[[y]]), 'regression', 'classification')

  # set engine based on algorithm
  engine = available_algorithms()[[algorithm]]
  model = engine(type)

  # set mode based on model type
  model = parsnip::set_mode(object = model, mode = type)

  # TODO implement possibility to feed in udf as evaluation metrics
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
      list(
        fold = 1,
        test = df,
        train = df
      )
    )
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
#' @param do_parallel bool, whether to perform \code{\link{score}} calls in parallel
#' @param n_cores numeric, number of cores to use, defaults to maximum minus 1
#'
#' @return a data.frame containing \describe{
#'     \item{x}{the name of the predictor variable}
#'     \item{y}{the name of the target variable}
#'     \item{result_type}{text showing how to interpret the resulting score}
#'     \item{pps}{the predictive power score}
#'     \item{metric}{the evaluation metric used to compute the PPS}
#'     \item{baseline_score}{the score of a naive model on the evaluation metric}
#'     \item{model_score}{the score of the predictive model on the evaluation metric}
#'     \item{cv_folds}{how many cross-validation folds were used}
#'     \item{seed}{the seed that was set}
#'     \item{algorithm}{text shwoing what algorithm was used}
#'     \item{model_type}{text showing whether classification or regression was used}
#' }
#'
#' @export
#'
#' @examples
#' \donttest{score_predictors(df = iris, y = 'Species')}
#' \donttest{score_predictors(df = mtcars, y = 'mpg', do_parallel = TRUE, n_cores = 2)}
score_predictors = function(df, y, ..., do_parallel = FALSE, n_cores = -1) {
  temp_score = function(x) {
    score(df, x = x, y = y, ...)
  }
  if (do_parallel) {
    if (n_cores == -1) {
      n_cores = parallel::detectCores() - 1
    }
    cl = parallel::makeCluster(n_cores)
    parallel::clusterExport(cl, varlist = as.list(.all_ppsr_functions()))
    scores = parallel::clusterApply(cl, colnames(df), temp_score)
    parallel::stopCluster(cl)
  } else {
    scores = lapply(colnames(df), temp_score)
  }
  scores = fill_blanks_in_list(scores)
  scores_df = do.call(rbind.data.frame, scores)
  rownames(scores_df) = NULL
  return(scores_df)
}


#' Calculate predictive power scores for whole dataframe
#' Iterates through the columns of the dataframe, calculating the predictive power
#' score for every possible combination of \code{x} and \code{y}.
#'
#' @inheritParams score
#' @inheritParams score_predictors
#'
#' @return a data.frame containing \describe{
#'     \item{x}{the name of the predictor variable}
#'     \item{y}{the name of the target variable}
#'     \item{result_type}{text showing how to interpret the resulting score}
#'     \item{pps}{the predictive power score}
#'     \item{metric}{the evaluation metric used to compute the PPS}
#'     \item{baseline_score}{the score of a naive model on the evaluation metric}
#'     \item{model_score}{the score of the predictive model on the evaluation metric}
#'     \item{cv_folds}{how many cross-validation folds were used}
#'     \item{seed}{the seed that was set}
#'     \item{algorithm}{text shwoing what algorithm was used}
#'     \item{model_type}{text showing whether classification or regression was used}
#' }
#' @export
#'
#' @examples
#' \donttest{score_df(iris)}
#' \donttest{score_df(mtcars, do_parallel = TRUE, n_cores = 2)}
score_df = function(df, ..., do_parallel = FALSE, n_cores = -1) {
  cnames = colnames(df)
  param_grid = expand.grid(x = cnames, y = cnames, stringsAsFactors = FALSE)
  temp_score = function(i) {
    score(df, x = param_grid[['x']][i], y = param_grid[['y']][i], ...)
  }
  if (do_parallel) {
    if (n_cores == -1) {
      n_cores = parallel::detectCores() - 1
    }
    cl = parallel::makeCluster(n_cores)
    parallel::clusterExport(cl, varlist = as.list(.all_ppsr_functions()))
    scores = parallel::clusterApply(cl, seq_len(nrow(param_grid)), temp_score)
    parallel::stopCluster(cl)
  } else {
    scores = lapply(seq_len(nrow(param_grid)), temp_score)
  }
  scores = fill_blanks_in_list(scores)
  df_scores = do.call(rbind.data.frame, scores)
  rownames(df_scores) = NULL
  return(df_scores)
}


#' Calculate predictive power score matrix
#' Iterates through the columns of the dataset, calculating the predictive power
#' score for every possible combination of \code{x} and \code{y}.
#'
#' Note that the targets are on the rows, and the features on the columns.
#'
#' @inheritParams score
#' @param ... any arguments passed to \code{\link{score_df}},
#'     some of which will be passed on to \code{\link{score}}
#'
#' @return a matrix of numeric values, representing predictive power scores
#' @export
#'
#' @examples
#' \donttest{score_matrix(iris)}
#' \donttest{score_matrix(mtcars, do_parallel = TRUE, n_cores=2)}
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
#' @param ... arguments to pass to \code{stats::cor()}
#'
#' @return a data.frame with x-y correlation coefficients
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
#' @param metric character, name of evaluation metric being used, see \code{available_evaluation_metrics()}
#'
#' @return numeric vector of length one, evaluation score for predictions using naive model
score_model = function(train, test, model, x, y, metric) {
  model = parsnip::fit(model,
                       formula = stats::as.formula(paste(y, '~', x)),
                       data = train)
  yhat = stats::predict(model, new_data = test)[[1]]
  eval_fun = available_evaluation_metrics()[[metric]]
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
#' @return numeric vector of length one, evaluation score for predictions using naive model
score_naive = function(train, test, x, y, type, metric) {
  eval_fun = available_evaluation_metrics()[[metric]]
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
#' @return numeric vector of length one, normalized score
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


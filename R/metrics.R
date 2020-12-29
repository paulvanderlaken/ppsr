
#' Generic evaluation function skeleton
#'
#' @param y vector of target values
#' @param yhat vector of predicted values
#'
#' @return NULL
.eval = function(y, yhat) {
  return(NULL)
}


#' Evaluate Mean Absolute Error
#'
#' @inheritParams .eval
#'
#' @return float scalar
#' @export
#'
#' @examples
#' y = c(1:10)
#' yhat = y + rnorm(n = length(y))
#' eval_mae(y, yhat)
eval_mae = function(y, yhat) {
  return(mean(abs(y - yhat)))
}

#' Evaluate Root Mean Squared Error
#'
#' @inheritParams .eval
#'
#' @return float scalar
#' @export
#'
#' @examples
#' y = c(1:10)
#' yhat = y + rnorm(n = length(y))
#' eval_rmse(y, yhat)
eval_rmse = function(y, yhat) {
  return(mean((y - yhat)^2)^0.5)
}


#' Calculate F1 scores
#'
#' @param y character or boolean vector of target values
#' @param yhat character or boolean vector of predicted values
#'
#' @return named vector of floats
calculate_f1_scores = function(y, yhat) {
  yhat <- factor(as.character(yhat), levels=unique(as.character(y)))
  y  <- as.factor(y)
  cm = as.matrix(table(y, yhat))

  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))

  # assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0

  return(f1)
}


#' Evaluate F1
#' Provides a wrapper around underlying functions that calculate either the
#' binary or the multiclass F1 scores, depending on the target data.
#' @inheritParams calculate_f1_scores
#' @param positive.class character scalar for the name of the positive class
#' @param average character scalar denoting the approach to average multiclass f1 scores
#'
#' @return float scalar representing the f1 score
#' @export
#'
#' @examples
#' y = letters
#' yhat = sample(y, replace = TRUE)
#' eval_f1_score(y, yhat)
#'
#' y = rbinom(100, 1, prob = 0.2)
#' yhat = sample(y, replace = TRUE)
#' eval_f1_score(y, yhat)
eval_f1_score = function(y, yhat, positive.class = '1', average = 'weighted') {
  # TODO add check for possible ways to calculate average
  if (length(unique(y)) > 2 | is.character(y)) {
    if (average == 'weighted') {
      return(eval_weighted_f1_score(y, yhat))
    } else if (average == 'macro') {
      return(eval_macro_f1_score(y, yhat))
    }
  } else if (is.integer(y) | is.logical(y) | is.factor(y)) {
    return(eval_binary_f1_score(y, yhat, positive.class))
  }
}

#' Evaluate binary F1
#' @inheritParams eval_f1_score
#'
#' @return float scalar representing the f1 score
#' @export
#'
#' @examples
#' y = rbinom(100, 1, prob = 0.2)
#' yhat = sample(y, replace = TRUE)
#' eval_binary_f1_score(y, yhat, positive.class = '1')
#'
#' y = as.logical(rbinom(100, 1, prob = 0.2))
#' yhat = sample(y, replace = TRUE)
#' eval_binary_f1_score(y, yhat, positive.class = '1')
eval_binary_f1_score = function(y, yhat, positive.class) {
  if (length(unique(y)) > 2) {
    warning('More than 2 unique target values detected. Consider using non-binary F1 formula.')
  }
  if (length(unique(yhat)) > 2) {
    warning('More than 2 unique target values detected. Consider using non-binary F1 formula.')
  }
  if (!positive.class %in% y) {
    warning('positive.class', positive.class, 'not detected in target values.')
  }
  f1 = calculate_f1_scores(y, yhat)
  return(f1[positive.class])
}

#' Evaluate macro F1 score (unweighted)
#'
#' @inheritParams calculate_f1_scores
#'
#' @return float scalar representing the f1 score
#' @export
#'
#' @examples
#' y = letters
#' yhat = sample(y, replace = TRUE)
#' eval_f1_score(y, yhat)
eval_macro_f1_score = function(y, yhat) {
  f1 = calculate_f1_scores(y, yhat)
  return(mean(f1))
}

#' Evaluate weighted F1 score
#'
#' @inheritParams calculate_f1_scores
#'
#' @return float scalar representing the f1 score
#' @export
#'
#' @examples
#' y = letters
#' yhat = sample(y, replace = TRUE)
#' eval_f1_score(y, yhat)
eval_weighted_f1_score = function(y, yhat) {
  f1 = calculate_f1_scores(y, yhat)
  w = prop.table(table(y))
  return(stats::weighted.mean(f1, w))
}

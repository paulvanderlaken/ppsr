

.eval = function(y, yhat) {
  return(NULL)
}

eval_mae = function(y, yhat) {
  return(mean(abs(y - yhat)))
}

eval_rmse = function(y, yhat) {
  return(mean((y - yhat)^2)^0.5)
}


calculate_f1_scores = function(y, yhat) {
  yhat <- factor(as.character(yhat), levels=unique(as.character(y)))
  y  <- as.factor(y)
  cm = as.matrix(table(y, yhat))

  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))

  f1[is.na(f1)] <- 0  # assuming that F1 is zero when it's not possible compute it

  return(f1)
}


eval_macro_f1_score = function(y, yhat) {
  f1 = calculate_f1_scores(y, yhat)
  return(mean(f1))
}

eval_weighted_f1_score = function(y, yhat) {
  f1 = calculate_f1_scores(y, yhat)
  w = prop.table(table(y))
  return(stats::weighted.mean(f1, w))
}

#' Lists all evaluation metrics currently supported
#'
#' @return a list of all available evaluation metrics and their implementation in functional form
#' @export
#'
#' @examples
#' available_evaluation_metrics()
available_evaluation_metrics = function() {
  return(list(
    'MAE' = eval_mae,
    'RMSE' = eval_rmse,
    'F1_macro' = eval_macro_f1_score,
    'F1_weighted' = eval_weighted_f1_score
  ))
}

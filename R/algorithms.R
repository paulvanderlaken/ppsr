


engine_tree = function(type) {
  return(parsnip::set_engine(parsnip::decision_tree(), "rpart"))
}

engine_glm = function(type) {
  if (type == 'regression') {
    return(parsnip::set_engine(parsnip::linear_reg(), "lm"))
  } else  if (type == 'classification') {
    return(parsnip::set_engine(parsnip::logistic_reg(), "glm"))
  }
}

engine_gbm = function(type) {
  return(parsnip::set_engine(parsnip::boost_tree(), "xgboost"))
}


#' Lists all algorithms currently supported
#'
#' @return a list of all available parsnip engines
#' @export
#'
#' @examples
#' available_algorithms()
available_algorithms = function() {
  return(list(
    'tree' = engine_tree,
    'glm' = engine_glm,
    'gbm' = engine_gbm
  ))
}

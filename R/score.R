score = function(x = mtcars$mpg, y = mtcars$cyl, algorithm = lm, metric = mae) {
  if (is_id(x) | is_id(y)) {
    # x or y is an id variable
    return(0)
  }
  if (is_constant(x) | is_constant(y)) {
    # x or y has no variance
    return(0)
  }
  if (is_same(x, y)) {
    return(1)
  }
  # store predictor and target in data frame
  df = data.frame(x = x, y = y)
  # drop any rows where target or predictor is missing
  df = df[complete.cases(df), ]
  fit = algorithm(y ~ x, data = df)
  yhat = predict(fit, newdata = df)
  score_original = metric(y = y, yhat = yhat)
  score_naive = metric(y = y, yhat = mean(y))
  score_normalized = 1 - score_original / score_naive
  return(score_normalized)
}

score_predictors = function(df, y) {
  scores = list()
  cnames = colnames(df)
  for (x in cnames) {
    scores[x] = score(x = df[[x]], y = df[[y]])
  }
  return(scores)
}

score_matrix = function(df) {
  cnames = colnames(df)
  mtrx = matrix(nrow = ncol(df), ncol = ncol(df), dimnames = list(cnames, cnames))
  for (target in cnames) {
    for (predictor in cnames) {
      mtrx[target, predictor] = score(x = df[[predictor]], y = df[[target]])
    }
  }
  return(mtrx)
}

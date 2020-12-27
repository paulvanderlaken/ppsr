library(ggplot2)

format_score = function(x, digits = 2) {
  return(round(x, digits = digits))
}

visualize_predictors = function(df, y) {
  predictors = score_predictors(df, y)
  df_scores = as.data.frame(predictors)
  features = colnames(df_scores)
  df_scores$.y = row.names(df_scores)
  row.names(df_scores) = NULL
  df_scores_long = tidyr::pivot_longer(df_scores,
                                       cols = tidyselect::all_of(features),
                                       names_to = '.x',
                                       values_to = 'score')
  p = ggplot(df_scores_long, aes(x = score, y = reorder(.x, score))) +
    geom_col(aes(fill = score)) +
    geom_text(aes(label = format_score(score)), hjust = 0) +
    scale_y_discrete(name = 'feature') +
    scale_fill_gradient(low = 'white', high = '#08306B') +
    theme_minimal()
  return(p)
}

visualize_matrix = function(df) {
  mtrx = score_matrix(df)
  df_scores = as.data.frame(mtrx)
  features = colnames(df_scores)
  df_scores$.y = row.names(df_scores)
  row.names(df_scores) = NULL
  df_scores_long = tidyr::pivot_longer(df_scores,
                                       cols = tidyselect::all_of(features),
                                       names_to = '.x',
                                       values_to = 'score')
  p = ggplot(df_scores_long, aes(x = .x, y = .y)) +
    geom_tile(aes(fill = score)) +
    geom_text(aes(label = format_score(score))) +
    scale_x_discrete(limits = features, name = 'feature') +
    scale_y_discrete(limits = rev(features), name = 'target') +
    scale_fill_gradient(low = 'white', high = '#08306B') +
    theme_minimal()
  return(p)
}

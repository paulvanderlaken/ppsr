#' Visualize the PPS of all predictors of a target
#'
#' @inheritParams score_predictors
#' @param y string, column name of target variable
#'
#' @return ggplot2 vertical barplot visualization
#' @export
#'
#' @examples
#' visualize_predictors(mtcars, 'mpg')
visualize_predictors = function(df, y) {
  predictors = score_predictors(df, y)
  df_scores = as.data.frame(predictors)

  # extract feature and target names
  features = colnames(df_scores)
  df_scores$.y = row.names(df_scores)
  row.names(df_scores) = NULL

  # transpose predictive power scores to tidy, long format for visualization
  df_scores_long = tidyr::pivot_longer(df_scores,
                                       cols = tidyselect::all_of(features),
                                       names_to = '.x',
                                       values_to = 'score')

  # visualize as heatmap
  p =
    ggplot2::ggplot(df_scores_long, ggplot2::aes(x = score, y = reorder(.x, score))) +
    ggplot2::geom_col(aes(fill = score)) +
    ggplot2::geom_text(aes(label = format_score(score)), hjust = 0) +
    ggplot2::scale_y_discrete(name = 'feature') +
    ggplot2::scale_fill_gradient(low = 'white', high = '#08306B') +
    ggplot2::theme_minimal()
  return(p)
}

#' Visualize the PPS matrix
#'
#' @param df data.frame
#'
#' @return ggplot2 heatmap visualization
#' @export
#'
#' @examples
#' visualize_matrix(mtcars)
visualize_matrix = function(df) {
  mtrx = score_matrix(df)
  df_scores = as.data.frame(mtrx)

  # extract feature and target names
  features = colnames(df_scores)
  df_scores$.y = row.names(df_scores)
  row.names(df_scores) = NULL

  # transpose predictive power scores to tidy, long format for visualization
  df_scores_long = tidyr::pivot_longer(df_scores,
                                       cols = tidyselect::all_of(features),
                                       names_to = '.x',
                                       values_to = 'score')
  p =
    ggplot2::ggplot(df_scores_long, ggplot2::aes(x = .x, y = .y)) +
    ggplot2::geom_tile(aes(fill = score)) +
    ggplot2::geom_text(aes(label = format_score(score))) +
    ggplot2::scale_x_discrete(limits = features, name = 'feature') +
    ggplot2::scale_y_discrete(limits = rev(features), name = 'target') +
    ggplot2::scale_fill_gradient(low = 'white', high = '#08306B') +
    ggplot2::theme_minimal()
  return(p)
}

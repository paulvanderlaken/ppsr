# solve warnings: no visible binding for global variable
.x <- .y <- NULL

#' Visualize the PPS of all predictors of a target
#'
#' @inheritParams score_predictors
#' @param y string, column name of target variable
#' @param color color used for highlighting high PPS
#'
#' @return ggplot2 vertical barplot visualization
#' @export
#'
#' @examples
#' visualize_predictors(iris, 'Species')
visualize_predictors = function(df, y, color = '#08306B', ...) {
  predictors = score_predictors(df, y, ...)
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
    ggplot2::ggplot(df_scores_long, ggplot2::aes(x = score, y = stats::reorder(.x, score))) +
    ggplot2::geom_col(ggplot2::aes(fill = score)) +
    ggplot2::geom_text(ggplot2::aes(label = format_score(score)), hjust = 0) +
    ggplot2::scale_y_discrete(name = 'feature') +
    ggplot2::scale_fill_gradient(low = 'white', high = color, limits = c(0, 1)) +
    ggplot2::expand_limits(fill = c(0, 1)) +
    ggplot2::theme_minimal()
  return(p)
}

#' Visualize the PPS matrix
#'
#' @inheritParams score_matrix
#' @inheritParams visualize_predictors
#'
#' @return ggplot2 heatmap visualization
#' @export
#'
#' @examples
#' visualize_matrix(iris)
visualize_matrix = function(df, color = '#08306B', ...) {
  mtrx = score_matrix(df, ...)
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
    ggplot2::geom_tile(ggplot2::aes(fill = score)) +
    ggplot2::geom_text(ggplot2::aes(label = format_score(score))) +
    ggplot2::scale_x_discrete(limits = features, name = 'feature') +
    ggplot2::scale_y_discrete(limits = rev(features), name = 'target') +
    ggplot2::scale_fill_gradient(low = 'white', high = '#08306B', limits = c(0, 1)) +
    ggplot2::expand_limits(fill = c(0, 1)) +
    ggplot2::theme_minimal()
  return(p)
}


pps_breaks = function() {
  return(seq(0, 1, 0.2))
}


#' Visualize the PPS of all predictors of a target
#'
#' @inheritParams score_predictors
#' @param y string, column name of target variable
#' @param color_pps color used for upper limit of PPS gradient (high PPS)
#' @param color_text color used for text, best to pick high contrast with \code(color_pps)
#'
#' @return ggplot2 vertical barplot visualization
#' @export
#'
#' @examples
#' visualize_predictors(iris, 'Species')
visualize_predictors = function(df, y, color_pps = '#08306B', color_text = '#000000', ...) {
  df_scores = score_predictors(df, y, ...)

  p =
    ggplot2::ggplot(df_scores, ggplot2::aes(x = pps, y = stats::reorder(x, pps))) +
    ggplot2::geom_col(ggplot2::aes(fill = pps)) +
    ggplot2::geom_text(ggplot2::aes(label = format_score(pps)), hjust = 0) +
    ggplot2::scale_x_continuous(breaks = pps_breaks(), limits = c(0, 1.05)) +
    ggplot2::scale_y_discrete(name = 'feature') +
    ggplot2::scale_fill_gradient(low = 'white', high = color_pps,
                                 limits = c(0, 1), breaks = pps_breaks()) +
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
visualize_matrix = function(df, color_pps = '#08306B', color_text = '#FFFFFF', ...) {
  cnames = colnames(df)
  df_scores = score_df(df, ...)

  p =
    ggplot2::ggplot(df_scores, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_tile(ggplot2::aes(fill = pps)) +
    ggplot2::geom_text(ggplot2::aes(label = format_score(pps)), col = color_text) +
    ggplot2::scale_x_discrete(limits = cnames, name = 'feature') +
    ggplot2::scale_y_discrete(limits = rev(cnames), name = 'target') +
    ggplot2::scale_fill_gradient(low = 'white', high = color_pps,
                                 limits = c(0, 1), breaks = pps_breaks()) +
    ggplot2::expand_limits(fill = c(0, 1)) +
    ggplot2::theme_minimal()
  return(p)
}

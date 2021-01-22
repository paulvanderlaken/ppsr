format_score = function(x, digits = 2) {
  return(formatC(x, format = 'f', digits = digits))
}

modal_value = function(x) {
  uq = unique(x)
  m = uq[which.max(tabulate(match(x, uq)))]
  return(m)
}


fill_blanks_in_list = function(ll) {
  elements_uq = unique(unlist(lapply(ll, names)))
  for (i in seq_along(ll)) {
    elements_filled = names(ll[[i]])
    elements_missing = setdiff(elements_uq, elements_filled)
    ll[[i]][elements_missing] = NA
  }
  return(ll)
}

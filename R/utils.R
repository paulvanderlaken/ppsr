format_score = function(x, digits = 2) {
  return(round(x, digits = digits))
}

modal_value = function(x) {
  uq = unique(x)
  m = uq[which.max(tabulate(match(x, uq)))]
  return(m)
}

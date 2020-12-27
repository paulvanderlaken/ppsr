
is_id = function(x) {
  return(is.character(x) & length(unique(x)) == length(x))
}

is_constant = function(x) {
  return(length(unique(x)) == 1)
}

is_same = function(x, y) {
  return(all(x == y))
}

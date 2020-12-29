
is_id = function(x) {
  return(is.character(x) & length(unique(x)) == length(x))
}

is_constant = function(x) {
  return(length(unique(x)) == 1)
}

is_same = function(x, y) {
  return(all(x == y))
}

is_binary = function(x) {
  return(length(unique(x)) == 2)
}

is_binary_numeric = function(x) {
  return(is_binary(x) & is.numeric(x))
}

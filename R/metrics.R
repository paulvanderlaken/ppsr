mae = function(y, yhat) {
  return(mean(abs(y - yhat)))
}

rmse = function(y, yhat) {
  return(mean((y - yhat)^2)^0.5)
}

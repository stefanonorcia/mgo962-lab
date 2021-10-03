library(fpp3)
ansett %>%
  filter(Airports == "MEL-SYD") %>%
  autoplot(Passengers)

library(fpp3)
ansett %>%
  filter(Class == "Economy") %>%
  autoplot(Passengers)

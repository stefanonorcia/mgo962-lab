library(fpp3)
tourism %>%
  summarise(Trips = sum(Trips)) %>%
  autoplot(Trips)

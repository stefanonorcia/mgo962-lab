library(fpp3)
ansett %>%
  select(Week,Airports,Class,Passengers) %>%
  summarise(total_passengers = sum(Passengers)) %>%
  autoplot(total_passengers)

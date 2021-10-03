library(fpp3)
holidays <- tourism %>%
filter(Purpose == "Holiday") %>%
  group_by(State) %>%
  summarise(Trips = sum(Trips))
holidays %>% autoplot(Trips)

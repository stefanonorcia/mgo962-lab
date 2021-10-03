library(fpp3)
holidays <- tourism %>%
filter(Purpose == "Holiday") %>%
  group_by(State) %>%
  summarise(Trips = sum(Trips))
holidays %>% gg_subseries(Trips) +
  labs(y = "thousands of trips",
    title = "Australian domestic holiday nights")
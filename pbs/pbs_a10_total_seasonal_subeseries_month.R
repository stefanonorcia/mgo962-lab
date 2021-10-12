library(fpp3)
PBS %>%
filter(ATC2 == "A10") %>%
select(Month, Concession, Type, Cost) %>%
  summarise(total_cost = sum(Cost)) %>%
  mutate(total_cost = total_cost / 1e6) %>%
  gg_season(total_cost, labels = "both") + labs(y = "$ million",
       title = "Seasonal plot: antidiabetic drug sales")
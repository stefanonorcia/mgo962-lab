library(fpp3)
beer <- aus_production %>%
select(Quarter, Beer) %>%
filter(year(Quarter) >= 1992)
beer %>% gg_season(Beer, labels="right")
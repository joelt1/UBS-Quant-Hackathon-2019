# UBS-Quant-Hackathon-2019
Group attempt at FX Value Strategy challenge

Case description:
1) Objective: To explore a trading strategy which take long/short position in G10 currencies (against USD) in order to extract value premium. In another words: to buy under-valued currency and to sell over-valued currency expecting price will revert to fair value.
2) Trading instrument: G10 FX forwards (can use 1w to 3month depending on trading frequency).
3) Trading frequency: Can be as frequent as once a day trading or less frequent.
4) Modeling inputs: FX spot/FX implied vol; rates; government bond yields in relevant jurisdiction, commodity price (oil, gold, etc); economic data (GDP, CPI etc) only if it is available.

Chosen strategy - Pairs Trading strategy:
1) Checks for unit roots on exchange rates to confirm non-stationarity of series
2) Searches for two cointegrated exchange rates satisfying p-value < 0.05 for stage 1 null hypothesis of no cointegration and p-value > 0.05 for stage 2 null hypothesis of at most 1 cointegrating equation
3) Once cointegrating pairs have been identified, establishes spread using normalised series
4) If last spread greater than pre-specified number of standard deviations (2 in back-test) based on rolling window for spread (30 days in back-test), simultaneously sells 1 week forward on overvalued currency and buys 1 week forward on undervalued currency
  Quotes are based on USD___ where ___ is the foreign currency
5) On the 7th day, reverses open positions by buying and selling overnight expiring forwards on overvalued and undervalued currencies respectively
6) Iterate for length of back-test

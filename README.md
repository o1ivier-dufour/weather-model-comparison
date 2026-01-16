# Hurricane Forecast Visual Comparison

This repository contains an early-stage prototype of an interactive application
designed to explore and analyze extreme weather events, with a particular focus
on hurricanes.

The core objective of the project is to **build the foundations for systematic
benchmarking of weather forecast models**, by comparing:
- observed atmospheric fields derived from reanalysis data (ERA5),
- and forecasts produced by different modeling approaches, including
  traditional numerical weather prediction models and more recent AI-based models.

At this stage, the work is **exploratory**. Visual analysis is used as a first step
to understand model behavior, identify meaningful variables, and highlight
typical failure modes (intensity errors, structural biases, timing issues, etc.)
before moving toward more formal quantitative benchmarks.

The long-term goal is to provide a robust framework in which multiple extreme
events (hurricanes, heatwaves, atmospheric rivers, etc.) can be analyzed in a
consistent way, eventually enabling reproducible and interpretable comparisons
between forecasting systems.

The current implementation focuses on a first case study: **Hurricane Dorian (2019)**.
It includes visualizations of wind and pressure fields, simple intensity indicators
such as the evolution of minimum surface pressure, and the initial building blocks
required for future trajectory and skill-score analyses.

---

## Demo (early prototype)

An early proof-of-concept version of the application is available here:

👉 [https://weather-model-comparison.streamlit.app/]

This demo represents only the first exploratory step of the project. Future
iterations will progressively introduce additional events, forecast datasets,
and quantitative evaluation metrics.



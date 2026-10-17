# Kenya Malaria Model — Drift Detection Dashboard

Production monitoring system that detects data drift in the Kenya malaria prediction model across three real-world scenarios using Evidently AI.

## Live Demo
https://model-monitoring-ez8ldvctczi67gypzrj48n.streamlit.app/

## What it monitors
Seasonal drift — rainy season increases parasite density across the lake endemic zone
Campaign drift — bed net distribution campaigns shift coverage rates significantly
Outbreak drift — new outbreaks bring younger patients with higher parasite loads

## Features
Drift Analysis tab — summary of drift results across all three scenarios
Feature Distributions tab — interactive comparison of baseline vs production distributions
Run Detection tab — live drift detection with Evidently AI and Plotly visualisations

## Tech Stack
Evidently AI · Plotly · Streamlit · Scikit-learn · Pandas · Python 3.11

GitHub: github.com/Mwauramos | Contact: mwauramos.n@gmail.com
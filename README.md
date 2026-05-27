# ECA Resilience

*[GitHub](https://github.com/worldbank/eca-resilience) · [Documentation](https://worldbank.github.io/eca-resilience/README.html) · [Development Data Partnership](https://datapartnership.org/)*

This project leverages large-scale GPS mobility data to study human activity and urban space usage in response to atypical events. Using the **Urban Space Usage Index**, a normalized measure of relative human presence derived from anonymized mobile device location pings, the project quantifies deviations from typical mobility patterns and examines how cities respond to a range of shocks, from sudden natural disasters to planned public events and slow-onset climate stressors.
 
Case studies span two countries and four distinct event types: the **2023 Turkey-Syria earthquakes**, **Republic Day celebrations in Istanbul**, **heatwaves in Metro Manila**, and **monsoon-driven flooding in Manila amplified by Typhoon Doksuri**.

## Project Overview
 
### Data Source
 
The analysis is based on the [Veraset Movement dataset](https://datapartnership.org/), provided as part of the Mobility Data collection from the [Development Data Partnership](https://datapartnership.org/). The dataset consists of anonymized, high-frequency GPS pings collected through a network of mobile applications and SDKs. Each record includes geographic coordinates, a UTC timestamp, and an anonymized device identifier.
 
Mobility observations are spatially aggregated using the [Uber H3 hierarchical spatial index](https://h3geo.org/) at resolutions 7 and 8 (average cell areas of ~5 km² and ~0.74 km², respectively).
 
### Methodology
 
The analytical framework follows three steps:
 
**1. Define a measure.** The **Urban Space Usage Index** (*I*) is defined as the daily share of total active users visiting each H3 hexagon, normalizing for day-to-day fluctuations in overall data volume:
 
> *I(h, d) = U(h, d) / U(d)*
 
where *U(h, d)* is the number of unique users in hexagon *h* on day *d*, and *U(d)* is the total number of active users on that day.
 
**2. Quantify deviations.** Deviations from typical conditions are measured through **Z-scores** computed relative to a stable baseline period:
 
> *Z(h, d) = (I(h, d) − μ(h)) / σ(h)*
 
**3. Interpret deviations.** Z-scores are analyzed temporally and spatially, and stratified by land-use category and functional layer (POI-based), enabling characterization of *where* and *how* urban activity changes in response to events.
 
Full methodological details are available in the [Methodological Framework](https://worldbank.github.io/eca-resilience/notebooks/Methodology.html) and [Spatial Characterization of Urban Units](https://worldbank.github.io/eca-resilience/notebooks/Methodology_land_usage.html) notebooks.
 
### Geographies
 
| Country | Area of Interest | Resolution |
|---|---|---|
| Philippines | Metro Manila | H3 resolution 8 (~0.74 km²) |
| Turkey | Istanbul | H3 resolution 8 (~0.74 km²) |
| Turkey | 11 earthquake-affected provinces | H3 resolution 8 (~0.74 km²) |
 
### Case Studies
 
A key design feature of this project is that the four case studies deliberately span **four distinct event typologies**, each presenting different challenges for mobility analysis and policy response:
 
| Event | Location | Typology | Period |
|---|---|---|---|
| [Republic Day](https://worldbank.github.io/eca-resilience/notebooks/Republic_day_report.html) | Istanbul, Turkey | Planned public event | Oct 2023 |
| [2023 Turkey-Syria Earthquake](https://worldbank.github.io/eca-resilience/notebooks/Earthquake_report.html) | Southern Turkey (11 provinces) | Sudden, unpredictable natural disaster | Feb 2023 |
| [Flooding (Typhoon Doksuri)](https://worldbank.github.io/eca-resilience/notebooks/Floods_report.html) | Metro Manila, Philippines | Foreseeable natural disaster (typhoon-driven) | Jul 2023 |
| [Heatwaves](https://worldbank.github.io/eca-resilience/notebooks/Heatwaves_report.html) | Metro Manila, Philippines | Slow-onset climate shock | Apr 2023 |
 
This typology-driven structure allows the framework to be tested across events with fundamentally different warning horizons, impact profiles, and behavioral responses:
 
- **Planned events** (Republic Day) produce sharp, predictable spikes in activity that amplify existing spatial patterns city-wide.
- **Sudden disasters** (earthquake) generate delayed but extreme anomalies driven by emergency response, displacement, and humanitarian operations, with no anticipatory behavioral signal.
- **Foreseeable disasters** (typhoon-driven flooding) show a characteristic two-phase pattern: a pre-event increase in activity consistent with anticipatory behaviors (stocking, relocation), followed by a sharp collapse during peak impact.
- **Slow-onset shocks** (heatwaves) produce weaker aggregate signals but reveal systematic spatial and functional redistributions of activity, with people shifting toward climate-controlled or shaded environments rather than reducing mobility altogether.
### Data Quality Assessments
 
Prior to analysis, comprehensive Exploratory Data Analysis and Quality Assessments (EDA+QA) were conducted for each country dataset, documenting temporal coverage, spatial distribution, regime shifts, and user-level heterogeneity.
 
| Report | Key findings |
|---|---|
| [EDA+QA Turkey](https://worldbank.github.io/eca-resilience/notebooks/Turkey_EDA_QA.html) | ~1.1B GPS points, 18.9M users; 96.7% temporal coverage; three anomalous regimes identified |
| [EDA+QA Metro Manila](https://worldbank.github.io/eca-resilience/notebooks/Manila_EDA_QA.html) | ~4.4B GPS points, 27.2M users; 96.6% temporal coverage; structural break on 10 July 2023 |



## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Lab for running notebooks

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/worldbank/eca-resilience.git
   cd eca-resilience
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate eca-resilience
   ```

### Usage

For detailed documentation and analysis notebooks, visit the [project documentation](https://worldbank.github.io/eca-resilience/README.html).

## Contact 

For questions, feedback, or contributions, please contact the Development Data Partnership at datapartnership@worldbank.org.

You can also open an issue in the [GitHub repository](https://github.com/worldbank/eca-resilience/issues).

## License

This project is licensed under the MIT License together with the World Bank IGO Rider. The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.

## Code of Conduct

This project maintains a [Code of Conduct](docs/CODE_OF_CONDUCT.md) to ensure an inclusive and respectful environment for everyone. Please adhere to it in all interactions within our community.

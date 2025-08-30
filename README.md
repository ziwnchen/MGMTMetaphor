# MGMTMetaphor

This repository contains the code for the **Management Metaphor Project**. In this project, we use two complementary approaches to measure management metaphors in text data:

- **Explicit Measurement:** Classify direct references to management-related words.
- **Implicit Measurement:** Utilize a fill-mask approach to quantify implicit metaphors that compare elements of private social life to economic and management concepts.

To test the robustness of our methods, we analyze five historical corpora:
- Movie Scripts
- Fiction
- The New York Times News
- Congressional Speeches
- Caselaw Opinions

## Project Pipeline

The project is structured into the following stages:

1. **Data Preprocessing**
   - **mgmt_preprocessing:** Cleans text data for explicit measurement.
   - **fillmask_preprocessing:** Cleans text data for implicit measurement.

2. **Feature Generation**
   - **explicit:** Generates features for the explicit measurement.
   - **implicit:** Generates features for the implicit measurement.

3. **Pooling**
   - **dataset_to_export_trend:** Combines data from all corpora to create the final dataset for trend analysis.
   - **dataset_to_export_agent:** Combines data from all corpora to create the final dataset for agent analysis.

4. **Trend Analysis**
   - **explicit_plot:** Plots the historical trend of explicit management metaphors.
   - **implicit_plot:** Plots the historical trend of implicit management metaphors.

5. **Agent Analysis**
   - Includes Stata commands for running marginal effect analysis.

Additionally, this repository provides the code and data used to train the classifiers employed during the feature generation process.

## Data Availability
The project uses five historical text corpora:
- New York Times Articles: The corpora could be accessed via [ProQuest TDM Studio](https://www.proquest.com/products-services/tdm-studio.html).
- Fiction: The corpora could be accessed via [HathiTrust Research Center Analytics](https://analytics.hathitrust.org/)
- Movie Scripts: The **publicly available** corpora is downloaded from [OpenSubtitles dataset](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles). At the time of analysis, we use the 2018 version.
- Congressional Speeches: The **publicly available** dataset is downloaded from [Stanford Congress Text](https://data.stanford.edu/congress_text) curated by Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy.
- Caselaw Opinions: The **publicly available** dataset is downloaded from [Caselaw Access Project](https://case.law/).

We also use one contemporary interview corpus:
- American Voice Project: https://americanvoicesproject.org/

Since some of the datasets are not publicly available, we provide access to the processed data used in our analysis through Google Drive:
https://drive.google.com/drive/folders/1ax3ICXA-8wEhVifqo9ZN1cBL5Y2sAeFQ?usp=sharing
Specifically, the "pooling" folder contains the processed data after the pooling stage, which can be used for trend and agent analysis. The "agent" folder contains results from marginal effect analysis which was used directly to produce the figures in the paper.
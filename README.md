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
   - **fillmask:** Generates features for the implicit measurement.

3. **Pooling**
   - **dataset_to_export_trend:** Combines data from all corpora to create the final dataset for trend analysis.
   - **dataset_to_export_agent:** Combines data from all corpora to create the final dataset for agent analysis.

4. **Trend Analysis**
   - **explicit_plot:** Plots the historical trend of explicit management metaphors.
   - **fillmask_plot:** Plots the historical trend of implicit management metaphors.

5. **Agent Analysis**
   - Includes Stata commands for running marginal effect analysis.

Additionally, this repository provides the code and data used to train the classifiers employed during the feature generation process.

---
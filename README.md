# MGMTMetaphor

This is the Code Repository of Management Metaphor Project. In this project, we use two approaches to measure management metaphor in text data. The first approach is to classifiy the explicit reference of management related words. The second approach is to use a fillmask approach to quantify implicit metaphors that compare objects in private social life to economic and management concepts. We use five different historical corpora to test the robustness of our measurement. The corpora are: movie scripts, fiction, nyt news, congressional speeches, and caselaw opinions.
 
The pipeline of the project is as follows:

1. Data Preprocessing
    - mgmt_preprocessing: cleaning text data for the explict measurement
    - fillmask preprocessing: cleaning text data for the implicit measurement

2. Feature Generation
    - explicit: generate features for the explicit measurement
    - fillmask: generate features for the implicit measurement

3. Pooling
    - dataset_to_export_trend: combine data from all corpora and generate the final dataset used in trend analysis
    - dataset_to_export_agent: combine data from all corpora and generate the final dataset used in agent analysis

4. Trend Analysis
    - explicit_plot: plot the historical trend of explicit management metaphor
    - fillmask_plot: plot the historical trend of implicit management metaphor

5. Agent Analysis
    stata commands used to run marginal effect analysis for the agent analysis

In addition, we also provide the code and data used when we train the classifiers used in the feature generation process.
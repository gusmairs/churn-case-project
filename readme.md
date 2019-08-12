### Customer churn prediction model

Uses real company dataset obtained by dsi (data not posted here) to enable prediction model development. The `/lib` directory has both `data_tools.py` and `model_tools.py`, with the file `dev/model_dev.py` driving the exploratory work. This project:
  - Incorporates some ETL processes focused on preparing a model matrix suitable for use in scikit-learn modeling processes
  - Is focused on prediction over inference, but follows some context analysis to use understanding of churn drivers to adjust and tune model features
  - Uses an 80-20 train-test split (split provided in the original dataset) and accuracy and ROC curve as standard analysis measures

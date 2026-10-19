Reproducibility Materials

This repository contains Python code for the manuscript “Foreclassing: From Forecasts to Decisions.”

Contents

MyModels.py: ForeClassNet architectures and baseline models

Train_models.py: model training and evaluation across five random seeds

Dataset scripts: data preparation for the energy, weather, and finance experiments

ScalableMultiRocketHydra.py: modified MultiRocketHydra implementation for the financial experiments

Train_stock_rocket.py: MultiRocketHydra training for the financial experiments

How to run

Install the required Python dependencies and run the relevant dataset and training scripts. The experiments use PyTorch and scikit-learn, with aeon used for MultiRocketHydra.

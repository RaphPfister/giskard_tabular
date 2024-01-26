# Homework for giskard

This project contains a ML project that aims at predicting the departure delay of American Airline.
This is more of a small prototype to quickly plug giskard library and hub into it than a heavy, industrial use-case.

See the related [medium article](https://medium.com/@raphael.pfister.00/debug-your-machine-learning-models-with-giskardai-a-quick-tour-on-model-robustness-993e383a2a35).

The `delay_predictor/` directory contains all the required sources to process the data (`Flight_delay.csv`) and train a tensorflow model.
Note that I didn't necessarily picked tensorflow for any particular reason, I just got inspiration from another model I initiated at my current position at Air France-KLM. That model uses `tensorflow_probability`'s `DistributionLayer` as a final layer to predict a conditional probability of delay, given the flight data.

`basic_pipeline_v1.ipynb` illustrates a workflow with a training pipeline and a few test cases with giskard in a jupyter notebook.

(feedbacks.md)[./feedback.md] contains my feedback after my experience with giskard.

## Get started

Create and activate a new conda environment using:

```bash
conda create -n giskard310 python=3.10
pip install -r requirements.txt
```

Run the whole data preparation + training pipeline:
```bash
python delay_predictor/featurizer.py
python delay_predictor/trainer.py --learning_rate=0.1 --epochs=10
```
This generates the model and the encoders as pickled files, as well as the train, val and test dataframes as parquet files.

You can can then run a (minimalistic) pytest suite that uses giskard to test the model behaviour against a metamorphic_increasing_test:
```bash
pytest tests/
```
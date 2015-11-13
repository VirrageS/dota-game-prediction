# Dota Game Prediction

It is Python script which predicts Dota's game result. Written for [Dota Prediction Challenge](https://www.hackerrank.com/challenges/dota2prediction/).

Score: **23.20** points (about **61.6%** correct predictions).

## About model

Script is using **Logistic Regression** from `sklearn.linear_model`. It predicts binary output using a weighted sum of predictor variables.
It is quite simple model and that is why it fails to capture any synergy or correlation between heroes.
This is probably the reason why global correct is only around 60%.

## Usage

	python3 dota.py < input_3000.in > tmp.out;

Then you can compare how different is our `tmp.out` from `output_3000.out`.

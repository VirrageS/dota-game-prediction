# Dota Game Prediction

It is Python script which predicts Dota's game result. Written for [Dota Prediction Challenge](https://www.hackerrank.com/challenges/dota2prediction/).

Score: **23.20** points (about **61.6%** correct predictions).

## About model

Script is using **Logistic Regression** from `sklearn.linear_model`.
It predicts binary output using a weighted sum of predictor variables.
It is quite simple model and that is why it fails to capture any synergy or correlation between heroes.
This is probably the reason why global correct is only around 60%.

You can also choose **SGDClassifier** (also from `sklearn.linear_model`) as model for this task.
Unfortunately this doesn't give you much more points - the scores are roughly the same as in **Logistic Regression**.

## Usage

	python3 dota.py < input_3000.in > tmp.out; diff -b tmp.out output_3000.out | wc -l;

This will tell you how different is your output `tmp.out` from `output_3000.out`.
The less, the better.

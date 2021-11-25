#!/usr/bin/env bash

python . --result-path paper-plots-results.csv --out overview-table.tex overview-table -gr --metrics ROC_AUC PR_AUC --dominant-metric ROC_AUC --aggregation mean

latexmk -pdf main.tex

# to cleanup all the tmp files:
# latexmk -c

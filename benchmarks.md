# Benchmarks

## Betti numbers

### 0

Trivial task
- GeneralAccuracy


### 1

Multi-class classification
- GeneralAccuracy
- MulticlassAccuracy

### 2

Binary classification

- GeneralAccuracy
- MulticlassAccuracy
- MCC

## Name

Multi-class classification with $N=5$

- $\text{Weighted Accuracy} = \frac{\sum_{i=1}^{N} w_i \cdot \text{Accuracy}_i}{\sum_{i=1}^{N} w_i}$
    - $ N $ is the number of classes.
    - $ w_i $ is the weight for class $i$, i.e. the number of instances of class $i$.
    - $ \text{Accuracy}_i $ is the accuracy for class $ i $, which is calculated as $ \frac{\text{True Positives}_i}{\text{Total Instances}_i} $.
- $\text{Balanced Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \text{Accuracy}_i$
    - $ N $ is the number of classes.
    - $ \text{Accuracy}_i $ is the accuracy for class $ i $, which is calculated as $ \frac{\text{True Positives}_i}{\text{Total Instances}_i} $.

## Orientability

Binary classification

- $\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- Matthews Correlation Coefficient 
    - $[-1, 1]$: 0 for no correlation and 1 for perfect correlation
    - $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$


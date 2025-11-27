## Model Performance Comparison

| Model              | Optimization   |   Accuracy |   Precision |   Recall |   F1-Score | Best Params                              |
|:-------------------|:---------------|-----------:|------------:|---------:|-----------:|:-----------------------------------------|
| KNN                | Initial        |     0.8730 |      0.8721 |   0.8730 |     0.8713 | N/A                                      |
| KNN                | Optimized      |     0.8889 |      0.8880 |   0.8889 |     0.8881 | {'n_neighbors': 9, 'weights': 'uniform'} |
| SVM                | Initial        |     0.8730 |      0.8721 |   0.8730 |     0.8713 | N/A                                      |
| SVM                | Optimized      |     0.8730 |      0.8755 |   0.8730 |     0.8729 | {'C': 10, 'kernel': 'linear'}            |
| RandomForest       | Initial        |     0.9206 |      0.9239 |   0.9206 |     0.9192 | N/A                                      |
| RandomForest       | Optimized      |     0.9365 |      0.9432 |   0.9365 |     0.9349 | {'max_depth': None, 'n_estimators': 50}  |
| LogisticRegression | Initial        |     0.8571 |      0.8571 |   0.8571 |     0.8543 | N/A                                      |
| LogisticRegression | Optimized      |     0.8889 |      0.8899 |   0.8889 |     0.8884 | {'C': 10}                                |
| NaiveBayes         | Initial        |     0.8254 |      0.8339 |   0.8254 |     0.8251 | N/A                                      |
| NaiveBayes         | Optimized      |     0.8254 |      0.8339 |   0.8254 |     0.8251 | N/A                                      |
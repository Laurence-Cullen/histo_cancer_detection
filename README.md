# histo_cancer_detection
Kaggle competition to tag lymph node biopsies as having tumorous cells or not.

https://www.kaggle.com/c/histopathologic-cancer-detection/

Experimenting with convolutional layers, batch norm with the top layer
composed of dense layers heavily dropped out.

Working to improve performance of data loader, currently somewhat random seeming
performance bottlenecks on train data load.

TODO Add output creation script, will require cycles of read write and model predictions

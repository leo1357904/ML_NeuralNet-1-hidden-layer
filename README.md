# ML_NeuralNet-1-hidden-layer
***Env: Python 3.6.9***

Neural Network for alphabet prediction.

### File Format ###
Each dataset (small, medium, and large) consists of two csv files—train and test. Each row contains 129 columns separated by commas. The first column contains the label and columns 2 to 129 represent the pixel values of a 16 × 8 image in a row major format. Label 0 corresponds to “a,” 1 to “e,” 2 to “g,” 3 to “i,” 4 to “l,” 5 to “n,” 6 to “o,” 7 to “r,” 8 to “t,” and 9 to “u.” Because the original images are black-and-white (not grayscale), the pixel values are either 0 or 1.




***neuralnet.py***: use generated data to predict the movie according to the movie description.

```
$ python neuralnet.py <train input> <test input> <train out> <test out> <metrics out> <epoch number> <hidden units number> <init flag> <learning rate>


$ python neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1
```

`<init flag>`: `1` for generating every weight with uniform distribution in [-0.1, 0.1] except bias term with 0. `2` for generating every weight 0.

`<train out>`: show predictions for every row data from `<train input>`.

`<test out>`: show predictions for every row data from `<test input>`.

`<metrics out>`: show detailed entropy for every epoch and the error rate for train data and test data.

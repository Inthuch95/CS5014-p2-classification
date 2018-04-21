# CS5014-p2-classification
## Parameter
### classifier type
-b = binary</br>
-m = multiclass</br>

### algorithm
logit = logistic regression </br>
tree = decision tree </br>
mlp = multi-layer perceptron </br>
svc = linear svm classification </br>


## Running preprocess.py
The preprocessing script can be found in src folder. It can be executed with the command: </br>
python preprocess.py model [classifier type] </br></br>
To run the train the preprocessing with binary dataset, you should use the command: </br>
python preprocess.py -b
## Running training_model.py
The model training script can be found in src folder. It can be executed with the command: </br>
python train_model.py model [classifier type] [algorithm] </br></br>
To run the train the model with logistic regression on binary dataset, you should use the command: </br>
python train_model.py -b logit
## Running final_model.py
The final model script can be found in src folder. It can be executed with the command: </br>
python final_model.py

## Notes
Graphviz and Pydotplus are required to visualise decision tree

## Solving Math Word Problems with Re-examination 
The code is the implementation of PseDual.

### Requirements
* Pytorch = 1.6.0;
* Transformers

### Data
Math23k and MathQA datasets can be downloaded from public sources and set to ./data directory. 

The annotations include problem tokens, expression, num_list and nums position in problem.

Pre-trained bert-base model for problems encoding can be downloaded to a ./pretrained_models directory.

### Train and Test
The main.py script can be used to train and test the model. 
```
python main.py
```

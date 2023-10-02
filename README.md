## Solving Math Word Problems with Re-examination 
The code is the implementation of PseDual for reexamination.

### Requirements
* Pytorch = 1.6.0;
* Transformers

### Data
Math23k and MathQA datasets can be downloaded from public sources and set to ./data directory. 

The annotations include problem tokens, expression, num_list and nums position in problem.

Pre-trained bert-base model for problems encoding can be downloaded to a ./pretrained_models directory.

### Train and Test
The main.py can be used to train and test model. The example trained model for math23k can be [downloaded](https://drive.google.com/file/d/1OZ7-6xx-dkOzGZWplwwqSOOHvpnYdGoR/view?usp=drive_link).
```
python main.py
```

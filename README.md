# EREI
Automatically Inserting Emojis in Responses for Open Domain Emotion-Controllable Dialogue Generation

## The Emojis Datas
We give the data related to the emoji used in the experiment:
1. Emoji Official Documentation: The file ```emoji-test.txt``` (http://unicode.org/Public/emoji/5.0/emoji-test.txt) provides data for loading and testing emojis. The 64 emojis that we used in our work are marked with '64' in our modified ```emoji-test.txt``` file.
    Unicode and the Unicode Logo are registered trademarks of Unicode, Inc. in the U.S. and other countries. 
    For terms of use, see http://www.unicode.org/terms_of_use.html
2. Emotag-1200: The emoji emotional vector we used in our model for quantification. 
3. Emoji2Vec: The emoji embedding representations (https://github.com/uclnlp/emoji2vec)

## Dependencies
* Python 3.5.2
* TensorFlow 1.2.1

## Dataset
1. Mojitalk: https://drive.google.com/file/d/1l0fAfxvoNZRviAMVLecPZvFZ0Qexr7yU/view?usp=sharing

## Train
### Dialogue generation
 1. Set the ```is_seq2seq``` variable in the ```cvae_run.py``` to ```True```
 2. Train the emoji classifier: ```CUDA_VISIBLE_DEVICES=0 python3 classifier.py``` 
 3. Train, test and generate: ```python3 rl_run.py```
### Type Selector
1. Add new  binary label for each response in the response dataset: ```python3 add_label.py```
2. train, test and classify:  ```python3 typeselctor.py```
### Generate final results
1. The Dialogue generation branch, the emoji branch and the type selector branch work together to get the final emotion-controllable emoji-rich responses: ```python3 response_with_emojis.py```.
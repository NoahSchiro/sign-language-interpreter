# American Sign Language Letter Interpreter
---

## Model
Model is sequential, built with Keras. Contains a combination of convolutional and dropout layers. Image input is a 64x64 image, and is classified into the letters A-Z, space, nothing, and delete. Model is compiled with adam optimizer, and the loss function categorical crossentropy. Model contains about 3 million trainable parameters.

Model is saved as an .h5 file and can be downloaded by itself if you just want the weights and biases (file size ~40MB).
### Improvements
I believe this accuracy can be improved if we reduce learning rate to 1e-4 and increased epochs to about 20-25. I don't currently have hardware to resonably acheive this.

### Dataset
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## How to run
### Requirements to run
Tensorflow, OpenCV, numpy, matplotlib (optional, but you will need to remove some code if you don't want to use this)

If you are training from scratch:
```
python3 train.py
```
This will place a train model in the models/ directory

Testing model:
```
python3 model_testing.py
```

## Future plans and improvements
- [ ] Run model with more epochs, and stochastic gradient decent instead of adam
- [ ] Try hyperparameter tuning
- [x] Add requirements.txt
- [ ] Implement an LSTM and try to recognize a limited set of gestures to expand beyond just letters
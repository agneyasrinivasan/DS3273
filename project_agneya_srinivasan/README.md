I was not able to upload the checkpoints folder, therefore I am giving this drive link to the checkpoints folder. 
[Link to the checkpoints](https://drive.google.com/drive/folders/1-filKxL8PxL5myJqNL2Ke5U6gwtEjHH_?usp=drive_link)

## Problem description

Coral reefs are vital marine ecosystems that support a diverse range of marine life. However, due to climate change, rising sea temperatures, and other environmental stressors, corals are increasingly experiencing bleaching. Coral bleaching occurs when corals expel the symbiotic algae living in their tissues, leading to a loss of colour and making them more vulnerable to mortality. The project will look at detecting whether the coral in the image is healthy or bleached. Example of images of healthy and bleached respectively, are given below:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXenPrqxT93_s6pGE4NvQQL-P6bkUG5vNoSXlHhsjcxYTciimCVoJOJW6tqDbpRb-sCNdWB6ydVfbQbTq3vc8TbudBYdaCsduntHXrkBptrUkDn0W8uxXaPHc282PE2YIi2BPz7lvw?key=EN8yFe6t2s2Aew0NvTTxiIAN)

## Problem formulation

Input: image of a coral (JPG format)
Output: Classification label indicating the state of the coral—healthy or bleached

## Data Source

Kaggle : [resource](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification)

## Model Architecture

*Input layer*
Image resized to (resize_x, resize_y, input_channels)

*Convolutional block 1*
- 2D Convolution Layer with 32 filters, kernel size 3×3, padding=1.
- Activation: ReLU
- MaxPooling Layer with kernel size 2×2

*Convolutional block 2*
- 2D Convolution Layer with 64 filters, kernel size 3×3, padding=1.
- Activation: ReLU
- MaxPooling Layer with kernel size 2×2

*Dropout layer*
Dropout with probability 0.25

*Fully connected layers*
- Flattening the feature maps into a 1D vector
- Fully Connected (Dense) Layer with 128 neurons
- Activation: ReLU
- Fully Connected (Dense) Layer with 'num_classes' output neurons

*Output layer*
Class scores for each class




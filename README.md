
The goal of this project is aimed at training a neural network on images using Fourier-based and Skip-connection-based neural network architectures.

SkipConn: Skip connections help mitigate the vanishing gradient problem and facilitate training of deeper networks. This model will directly learn from the pixel values of the image. While the skip connections provide a form of shortcut for the gradients and allow for better information flow, the model will noticaebly struggle to capture periodic patterns or intricate details without extensive training or additional layers.

Fourier: By first transforming the image data into Fourier space, this model can leverage the strength of frequency domain representation. The intricate details and periodic patterns of the image, which might be challenging to capture in the original space, can be more easily represented and learned in the Fourier space. Once the Fourier features are computed, they are passed through the SkipConn model, combining the strengths of both Fourier transformation and skip connections.

CenteredLinearMap: A utility class to linearly scale and translate the data.

The model leverages the idea of skip connections. For a given layer, instead of just passing the output of the previous layer, we combine the outputs of one or more preceding layers. 
This can be represented as:

![image](https://github.com/samjsnn/Image-Training-Neural-Network/assets/106383967/fc7a2ecd-e0a8-4e4d-ba8a-b067d48d74f9)

Fourier Features: The Fourier model expands the input features using both sine and cosine functions of different orders. This aids the model in capturing more complex patterns and periodicities in the data. 

The Fourier transformation is mathematically representd as:

![image](https://github.com/samjsnn/Image-Training-Neural-Network/assets/106383967/ce4ca45a-8ca0-47d7-ad60-8e49a67b402c)


Loss Function: The Mean Squared Error (MSE) loss function is used, which is given by:

![image](https://github.com/samjsnn/Image-Training-Neural-Network/assets/106383967/533e2777-164e-4a3b-a236-7f8a80caed48)


Optimizer: The Adam optimizer is used for training, which is an adaptive learning rate optimization algorithm. It computes adaptive learning rates for each parameter.

Learning Rate Scheduler: The learning rate is decreased by half every 3 epochs using a step scheduler.


Ensure CUDA is available for GPU acceleration. Otherwise, remove the .cuda() calls and train the model on CPU.

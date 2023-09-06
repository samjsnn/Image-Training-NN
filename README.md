
This project is aimed at training a neural network on images using Fourier-based and Skip-connection-based neural network architectures.

SkipConn: This architecture uses linear layers with skip connections between them. Skip connections help mitigate the vanishing gradient problem and facilitate training of deeper networks.
Fourier: This architecture employs Fourier features before feeding the data into the inner model, in this case, the SkipConn model. Fourier features can help the model generalize better to different frequencies in the data.
CenteredLinearMap: A utility class to linearly scale and translate the data.

The model leverages the idea of skip connections. For a given layer, instead of just passing the output of the previous layer, we combine the outputs of one or more preceding layers. 
This can be represented as:
![image](https://github.com/samjsnn/Image-Training-Neural-Network/assets/106383967/fc7a2ecd-e0a8-4e4d-ba8a-b067d48d74f9)

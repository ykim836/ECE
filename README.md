# A Convolutional Approach to Quality Monitoring for Laser Welding

## The project purpose 
> This project purposes a stable quality monitoring system for the laser welding process using convolutional neural network.

## Laser-Welding process
> The Laser-Welding process consists of two steps; pre-welding and post-welding.  
Before welding, the detector should find each surface of hairpins apart from the background.  
Also, it should find the center of the surface, which would be the target point for the laser beam.  
After welding, the inspector classifies all the classes of welding.  
The general features of incorrectly welding are 'Lack of fusion', 'Adhered spatter', 'Unwelded', 'Blow out', 'Overwelded', and 'Misaligned'.  
Based on the features from quality deviations, it should filter the incorrectly welded hairpins from the correctly welded ones.  
This helps to detect weld failures before they get further processed inline.
<img src = "https://user-images.githubusercontent.com/81494108/125701722-8d3ea349-a118-475f-9029-b57957a577d6.png" width="70%" height="70%">  

## Small Dataset analysis
>The data file has two data files, which are train and test files.  
>In the train file, 18 images exist in each normally and abnormally welding file. In the test file, 9 images exist in each case.  

## Code analysis

>### Base-Line model  
>1. Load the data  
>The images of the original file should be accumulated into a dataset.  
>First, original images were assigned with paths, and then, saved as a type of array.  
>get_data function loads the data into dataset, convert images format with BGR to RGB and resize the images to all identical sizes.  
>
>2. Check the number of images in both classes & Visualize random images  
>Using the for loop, the number of images of both classes in train and validation files was checked.  
>By visualizing the images, the output from the end of this training could be expected.
>
>3. Data Preprocessing  
>After making the train and test dataset with feature and label, the dataset should be normalized.    
>By deviding each array by 255 and reshaping the images into initial defined size, the whole dataset was normalized.  
>Using the ImageDataGenerator of the Keras library, the dataset was augmented by zooming, flipping, shift, rotation.   
>The mean and std was not changed from the original dataset.
>
>4. Build a CNN model  
>The structure of the CNN model has four convolutional layers, which are followed by MaxPool layer.  
>From 8by8 (the first filter size), the following filter size was doubled reaching to 64by64.  
>There are no padding, and the activation function was ReLU.  
>At the last layer, the dropout layer was added.
>
>5. Train and Test with model  
>Using the Adam optimizer, the model was trained with a learning rate of 0.000001. The 500 epochs were performed.  
>
>6. Result discussion  
>From the model fit, the loss and accuracy of each train and validation data learning results were obtained.  
>Using the classification_report, the final chart of loss and accuracy could be inspected.

>### VGG blocks applied models  
>For better image classification, the Base-Line model architecture has been transformed.  
>The basic architecture of VGGNet has groups of convolutional layers that use small filters followed by the max-pooling layer.  
>Refer to the structure of the VGGNet, the specification of the VGG block can be generalized.  
>There are one or more convolutional layers with the same number of filters and a filter size of 3by3, stride of 1by1, and same padding.  
>The following max-pooling layer has a size of 2by2 and stride of the same dimensions.
>
>Applying the VGG blocks, 1 VGG block model and 3 VGG blocks model were achieved.
>The 1 VGG block model has 1 VGG block which contains two convolutional layers and one max-pooling layer.  
>The 3 VGG blocks model has 3 VGG blocks and each block has different number of kernels in the convolutional layers.  

>### VGG blocks with overfitting prevention models  
>The batch normalization is a technique designed to automatically standardize inputs to alayer in deep learning neural network.
>This batch normalization layer is located between the convolutional layer and the following max-pooling layer.  
>The L2 regularization adds a penalty when the model complexity increases.  
>The regularization parameter, lambda, penalizes all the parameters except intercept so that the model generalizes the data without overfitting.  

## Note  
>You can see more details in the "final_project_report_YeongeunKim.pdf".  
>The small dataset "two_class_post_weld.zip" can be downloaded in this repository.  
>After download it, please change the code lines, which are paths to the both train and test folders.  
>For the Base-Line model, the code file is "baseline.py".  
>With VGG blocks applied, the code files are "1blockvgg.py" and "3blockvgg.py".  
>With the overfitting prevention methods, the code files are "baseline_overfittingprevention.py", "1blockvgg_overfittingprevention.py" and "3blockvgg_overfittingprevention.py".  

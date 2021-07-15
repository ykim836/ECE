# A Convolutional Approach to Quality Monitoring for Laser Welding

## The project purpose 
> This project purposes a stable quality monitoring system for the laser welding process using convolutional neural network.

## Laser-Welding process
> The Laser-Welding process consists of two steps; pre-welding and post-welding.  
Before welding, the detector should find each surface of hairpins apart from the background.  
Also, it should find the center of the surface, which would be the target point for the laser beam.  
After welding, the inspector classifies all the classes of welding; normally welded and abnormally welded states.  
The general features of incorrectly welding are 'Lack of fusion', 'Adhered spatter', 'Unwelded', 'Blow out', 'Overwelded', and 'Misaligned'.  
Based on the features from quality deviations, it should filter the incorrectly welded hairpins from the correctly welded ones.    
<img src = "https://user-images.githubusercontent.com/81494108/125701722-8d3ea349-a118-475f-9029-b57957a577d6.png" width="70%" height="70%">  

## Post-Welding process

>### Small Data analysis  
>The original data file has two data files, which are train and test files.  
>In the train file, 18 images exist in each normally and abnormally welding file. In the test file, 9 images exist in each case.  

>### Code analysis

>1. Load the data  
>The images of the original file should be accumulated into a dataset.  
>First, original images were assigned with paths, and then, saved as a type of array.  
>get_data function loads the data into the dataset, convert images format with BGR to RGB and resize the images to all identical sizes.  
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
>There are no padding, and the activation function was relu.  
>At the last layer, the dropout layer was added.
>
>5. Train and Test with model  
>Using the Adam optimizer, the model was trained with a learning rate of 0.000001. The 500 epochs was performed.  
>
>6. Result discussion. 
>From the model fit, the loss and accuracy of each train and validation dataset learning results were obtained.  
>Using teh classification_report, the final chart of loss and accuracy could be inspected.

## Note
>The code file is "post_welding_code.py".  
>The small dataset "two_class_post_Weld.zip" can be downloaded in this repository.  
>After download it, please change the code lines, which are paths to the both train and test folders.  
>Using your GoogleColaboratory, this program can be finished in 30 seconds.  

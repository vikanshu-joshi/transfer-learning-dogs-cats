# transfer-learning-dogs-cats
VGG16 model using transfer learning on Dogs vs Cats classifier

Dogs vs Cats: <a href="https://www.kaggle.com/c/dogs-vs-cats">kaagle</a><br>
The problem statement is, given an image, identify whether it is a dog or a cat image.

## Run the script
<li>To directly runt he model on images, move the files "prediction.py" & "trained_model" in directory with all images.
<li>Then run python prediction.py using powershell/terminal
<li>script will move images into directories name dogs, cats respectively
<li>script will process images in batch of 1000 to avoid run out of memory issues
<li>see demo video <a href="https://drive.google.com/file/d/1LDqMVD46IgtR7a0_un1eGTiUNqBjJvaE/view?usp=sharing">here</a>

## jupyter notebook
<li>model.ipynb/model.html contains code of model with train test and validation
<li>The model is trained on 8000 images 4000 on each class.
<li>Then it is tested on 200 images and output is printed alongwith image.
<li>Model reached an accuracy of 99% in 3 epochs.

 ## predictions preview
 <img src="https://github.com/vikanshu-joshi/transfer-learning-dogs-cats/blob/master/preview.png"/>

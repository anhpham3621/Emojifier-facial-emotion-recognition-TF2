Note:
all dependencies in requirements.txt are not imported likely due to version incompatibility
you should manually install using pip on MacOS
pip will download the latest version

edit config.ini to customize your local paths

python 3.9 or 3.10 should work 

virtual environment recommended for your local machine 


Key Features of haarcascade_frontalface_default.xml:
Purpose: It helps identify regions in an image that likely contain a human face.
Implementation: The model uses Haar-like features, which are pre-defined patterns of intensity (light and dark regions), combined with machine learning for classification.
Training: The classifier is trained using thousands of positive and negative images to detect faces effectively.

**Link to working Google Drive:**
https://drive.google.com/drive/folders/1alpsE57XJWarzPfAYzLK9wbYFTA7Wlz4?usp=sharing

**Link to source code:**
https://github.com/vijuSR/facial_emotion_recognition__EMOJIFIER

**Setup Instructions**
**1. Create a Virtual Environment**

Create a virtual environment using the following command:

python3 -m venv venv

**2. Activate the Virtual Environment**

**Linux/macOS:**

source venv/bin/activate

**Windows:**

cd venv

.\Scripts\activate

**3. Install Dependencies**
Install the following packages manually (to avoid version incompatibilities):

pip install --upgrade pip

pip show opencv-python  # If none, install it below

pip install opencv-python

pip install tensorflow-macos # if use MacOS

pip install tensorflow-metal # if use MacOS

pip install tqdm

pip install FuzzyTM  # Required by other dependencies

pip install "pyqt5<5.16"  # Version constraint due to compatibility

pip install numpy==1.24.4  

# Ensure compatibility with source code, note there might be other versions that also fit
**Explanation:**

tensorflow-macos: Enables TensorFlow on macOS (supports CPU-based operations).

tensorflow-metal: Provides GPU acceleration using Apple’s Metal API.

pyqt5<5.16 and FuzzyTM: Suggested during installation for resolving dependency issues.

**4. Configure Paths**

Run the following supporting script to get the path to the Haar Cascade file:

python3 get_haarcascade_path.py

Copy the displayed path and update it in config.ini under the haarcascade_path field.

**Run Instructions**

**STEP 0 - define your EMOTION-MAP**

cd <to-repo-root-dir>

Open the 'emotion_map.json'
Change this mapping as you desire. You need to write the "emotion-name". Don't worry for the numeric-value assigned, only requirement is they should be unique.
There must be a .png emoji image file in the '/emoji' folder for every "emotion-name" mentioned in the emotion_map.json.
Open the 'config.ini' file and change the path to "haarcascade_frontalface_default.xml" file path on your system. For example on my system it's: > "G:/VENVIRONMENT/computer_vision/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml" where > "G:/VENVIRONMENT/computer_vision" is my virtual environment path.
'config.ini' contains the hyperparameters of the model. These will depend on the model and the dataset size. The default one should work fine for current model and a dataset size of around 1.2k to 3k. IT'S HIGHLY RECOMMENDED TO PLAY AROUND WITH THEM.

**STEP 1 - generating the facial images**

cd </to/repo/root/dir>

run python3 src/face_capture.py --emotion_name <emotion-name> --number_of_images <number>

-- example: python3 src/face_capture.py --emotion_name smile --number_of_images 200

This will open the cam and all you need to do is give the smile emotion from your face.
NOTE: You must change /emotion_map.json if you want another set emotions than what is already defined.
Do this step for all the different emotions in different lighting conditions.
For the above result, I used 300 images for each emotions captured in 3 different light condition (100 each).
You can see your images inside the 'images' folder which will contain different folder for different emotion images.

**STEP 2 - creating the dataset out of it**

run python3 src/dataset_creator.py

This will create the ready-to-use dataset as a python pickled file and will save it in the dataset folder.

**STEP 3 - training the model on the dataset and saving it**

run python3 src/trainer.py

This will start the model-training and upon the training it will save the tensorflow model in the 'model-checkpoints' folder.
It has the parameters that worked well for me, feel free to change it and explore.

**STEP 4 - using the trained model to make prediction**

run python3 src/predictor.py

Tip: you can output the predictor results into a log file for convenience (instead of having the results printed in terminal which can get burdensome): 
python3 src/predictor.py > output.log 2>&1
this will open the cam, and start taking the video feed 


Step 1 : pip install -r requirements.txt
Step 2 : To load the dataset follow these steps
        --> pip install kaggle
        --> Go to your Kaggle account.
        --> Navigate to Account settings by clicking on your profile picture.
        --> Scroll down to the API section and click on "Create New API Token". This will download a kaggle.json file.
        --> Move the kaggle.json file to the appropriate location:
        --> Linux/Mac: ~/.kaggle/kaggle.json
        --> Windows: C:\Users\<Your Username>\.kaggle\kaggle.json
        --> kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
Step 3 : Keep the unziped dataset inside the same directory as the model.py file

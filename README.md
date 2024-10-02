# CIFake: Real and AI-Generated Synthetic Images

This repository contains code to work with the CIFake dataset, which includes real and AI-generated synthetic images. Follow the steps below to set up the project and load the dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [Usage](#usage)

## Installation

1. Clone the repository and navigate to the project directory

2. Install the required packages by running:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset Download

To download the CIFake dataset from Kaggle, follow these steps:

1. Install the Kaggle package:

    ```bash
    pip install kaggle
    ```

2. Go to your [Kaggle account](https://www.kaggle.com/).

3. Navigate to **Account Settings** by clicking on your profile picture in the top-right corner.

4. Scroll down to the **API** section and click on **Create New API Token**. This will download a `kaggle.json` file to your system.

5. Move the `kaggle.json` file to the appropriate location based on your operating system:

    - **Linux/Mac**: `~/.kaggle/kaggle.json`
    - **Windows**: `C:\Users\<Your Username>\.kaggle\kaggle.json`

6. Download the CIFake dataset:

    ```bash
    kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
    ```

7. Unzip the downloaded dataset and place it inside the same directory as the `model.py` file.

## Usage

Once you have the dataset and environment set up, you can run the provided model to train and test on the CIFake dataset.

```bash
python model.py

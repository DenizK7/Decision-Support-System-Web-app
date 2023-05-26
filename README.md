Welcome to my project, where you can train a machine learning model and make predictions using a web app built with React and Django.
Overview
This app allows you to upload files with the extensions .model or .arff, which are then stored in a database. You can choose from the uploaded files at any time, and if you select a file with the .arff extension and click the train button, the app will train the file with the dataset in the background and create a new file with the .model extension.

# Machine Learning Web App

## Overview

This web app allows you to upload machine learning files with the extensions `.model` or `.arff`. The uploaded files are stored in a database, and you can choose from the uploaded files at any time.

If you select a file with the `.arff` extension and click the train button, the app will train the file with the dataset in the background and create a new file with the `.model` extension.

Once you have a trained model, you can enter data according to the dataset and use the predict button to make predictions and see the results. If you upload a pre-trained `.model` file, you can skip the training step and directly use the predict button to make predictions.

## Technologies Used

This web app was built using the following technologies:

- React
- Django
- SQLite
- JavaScript
- HTML
- CSS

## Installation

To run the web app on your local machine, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `npm install` and `pip install -r requirements.txt`.
3. Set up a SQLite database and update the `DATABASES` setting by running `python manage.py makemigrations` then `python manage.py migrate`.
4. Run the Django server using the command `python manage.py runserver`.
5. In a separateterminal window, run the React app using the command `npm start`.

## Usage

To use the web app, follow these steps:

1. Upload a file with the extension `.model` or `.arff`.
2. Choose a file from the uploaded files list.
3. If you selected a file with the `.arff` extension, click the train button to train the file with the dataset.
4. If you uploaded a pre-trained `.model` file, skip the training step and directly use the predict button to make predictions.
5. Once the training is complete or you have uploaded a pre-trained `.model` file, enter data according to the dataset and click the predict button to make predictions.
6. View the results of your predictions.

## Contribution Guidelines

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository and create a new branch for your changes.
2. Make your changes and commit them to your branch.
3. Test your changes thoroughly.
4. Create a pull request with a clear description of your changes and why they are necessary.

## Credits

This project was created by Deniz Kucukkara. Special thanks to any contributors or resources used.

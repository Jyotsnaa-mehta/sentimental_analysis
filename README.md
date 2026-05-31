# Fake Product Review Detection System

## Introduction

Online shopping has become a major part of our daily lives, and customer reviews play an important role in helping people make purchasing decisions. However, many products receive fake reviews that can mislead customers. This project aims to detect whether a product review is genuine or fake using Machine Learning techniques.

## About the Project

The Fake Product Review Detection System analyzes the text of a product review and predicts whether it is genuine or fake. The application uses Natural Language Processing (NLP) and Machine Learning algorithms to process review data and make predictions.

A simple and interactive user interface has been developed using Streamlit, allowing users to enter a review and instantly view the prediction result.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization

## Machine Learning Algorithms

The following algorithms were used for training and evaluation:

- Logistic Regression
- Naive Bayes

## Working of the System

1. The user enters a product review.
2. The review text is cleaned and preprocessed.
3. TF-IDF Vectorization converts the text into numerical features.
4. The trained machine learning model analyzes the review.
5. The system predicts whether the review is Fake or Genuine.
6. The result is displayed on the screen.

## Features

- Detects fake and genuine reviews
- Simple and user-friendly interface
- Fast prediction results
- Uses machine learning for classification
- Helps improve trust in online reviews

## How to Run the Project

1. Download or clone the repository.
2. Install the required libraries using:

pip install -r requirements.txt

3. Run the application using:

streamlit run app.py

4. Open the local URL displayed in the terminal and start testing reviews.

## Future Improvements

- Improve prediction accuracy using larger datasets
- Implement Deep Learning models
- Add support for multiple languages
- Integrate with e-commerce platforms
- Provide detailed review analysis

## Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can be used to identify fake product reviews. It provides a simple solution for detecting misleading reviews and can be further enhanced for real-world applications.

## Author

Jyotsna Mehta  
Bachelor of Computer Applications (BCA)

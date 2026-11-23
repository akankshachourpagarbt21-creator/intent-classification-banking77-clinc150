# Comparative Study of Machine Learning and Deep Learning Approaches for Chatbot Intent Classification

## Overview
This project compares classical Machine Learning, task-trained Neural Network, and pretrained Transformer models for intent classification in chatbots.

The study evaluates model performance under different training data sizes to understand the trade-off between classification accuracy, computational efficiency, and model size.

## Models

### Classical Machine Learning
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

### Neural Networks
- BiLSTM
- TextCNN

### Transformer
- DistilBERT

## Datasets
- **Banking77** – Banking-related customer queries across 77 intent classes.
- **CLINC150** – Intent classification dataset covering multiple domains.

## Evaluation
The study considers:
- Accuracy
- Macro-F1 score
- Training time
- CPU inference latency
- Model size
- Performance with different training data sizes

Experiments are designed to use multiple training data sizes and random seeds.

## Project Status
The code and experimental setup are being organized. The experiments are currently being rerun after the previous generated output files were accidentally lost. Final results, tables, and plots will be updated once the experiments are completed and verified.

## Repository Contents
Training scripts, preprocessing utilities, setup files, and analysis code will be added and organized in this repository.

## Authors
Akanksha Chourpagar  
Dhananjay Potinda

MCA (Data Science)  
Dr. D.Y. Patil Institute of Management and Entrepreneur Development, Talegaon, Pune, India

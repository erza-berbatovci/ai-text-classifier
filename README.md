# ai-text-classifier


##  Project Overview  
This project demonstrates a complete text classification pipeline using the [AG News dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).  
The task is to classify news articles into four categories:  
-  World  
-  Sports  
-  Business  
-  Sci/Tech  

---

##  Features  
- Data loading & preprocessing (cleaning, stopword removal, lowercasing)  
- Exploratory analysis (class distribution, frequent words)  
- Model training with Logistic Regression and Naive Bayes  
- Model evaluation (accuracy, precision, recall, F1-score)  
- Prediction function for new text inputs  
- Visualization: class distribution and confusion matrix  

---

## Dataset  
- Source: [Kaggle – AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
- ~127k samples, evenly distributed across 4 categories
---

## How to Run
# 1. Clone the repository
git clone [https://github.com/erza-berbatovci/ai-text-classifier.git]
cd AI_task

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training script
python ag_news.py

# 4. Make predictions
python predict.py "Your text here


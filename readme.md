# California House Price Predictor

A complete end-to-end Machine Learning project built using the California Housing dataset from Hands-On Machine Learning (HOML).

This project demonstrates the full machine learning workflow — from raw data ingestion and preprocessing to model training, hyperparameter tuning, evaluation, and deployment preparation.

---

## Project Overview

The goal of this project is to predict median house values across California districts using supervised machine learning.

The pipeline includes:

* Data collection and extraction
* Exploratory Data Analysis (EDA)
* Data visualization
* Feature engineering
* Missing value handling
* Custom preprocessing transformers
* Feature scaling and encoding
* Model training
* Cross-validation
* Hyperparameter optimization
* Final model evaluation
* Prediction generation

---

## Dataset

The project uses the **California Housing Dataset**.

Features include:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity

Target Variable:

**Median House Value**

---

## Tech Stack

### Languages

* Python

### Libraries Used

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SciPy
* Joblib

---

## Machine Learning Workflow

### 1. Data Loading

* Automated dataset download
* Extraction from compressed archive
* CSV parsing

---

### 2. Exploratory Data Analysis

Performed:

* Dataset inspection
* Statistical summary
* Distribution analysis
* Correlation analysis
* Scatter plots
* Geographical data visualization
* Income category stratification

---

### 3. Data Preprocessing

Implemented:

#### Missing Value Handling

Median-based imputation

#### Feature Scaling

* Standard Scaling
* Min-Max Scaling

#### Categorical Encoding

* One-Hot Encoding
* Ordinal Encoding

#### Feature Engineering

Created derived features such as:

* Rooms per household
* Bedrooms ratio
* Population per household

---

### 4. Custom Transformers

Built custom preprocessing components including:

#### ClusterSimilarity Transformer

Geospatial similarity transformation using:

* KMeans clustering
* RBF kernel similarity

#### Ratio Pipeline

Custom feature ratio generation

---

### 5. Model Training

Trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

### 6. Hyperparameter Optimization

Implemented:

#### Grid Search

Exhaustive parameter search

#### Randomized Search

Efficient randomized hyperparameter sampling

Optimized parameters included:

* Number of geographical clusters
* Random forest feature selection depth

---

### 7. Model Evaluation

Evaluation metrics used:

* Root Mean Squared Error (RMSE)
* Cross Validation
* Bootstrap Confidence Intervals

Final Model Performance:

**RMSE ≈ 41,445**

---

## Project Structure

```bash
California-House-Price-Predictor/
│
├── datasets/
│   └── housing/
│
├── california_house_predictor.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/california-house-price-predictor.git
cd california-house-price-predictor
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Launch Notebook

```bash
jupyter notebook
```

Open:

`california_house_predictor.ipynb`

---

## Deployment

This project is deployment-ready and can be served using:

* Streamlit
* FastAPI
* Flask

---

## Key Learning Outcomes

This project demonstrates practical understanding of:

* End-to-end ML pipeline construction
* Production-style preprocessing
* Feature engineering
* Custom Scikit-learn transformers
* Model serialization
* Hyperparameter tuning
* Performance evaluation
* Prediction serving

---

## Future Improvements

Planned enhancements:

* Interactive web interface
* API deployment
* Model monitoring
* Feature importance dashboard
* Real-time prediction service
* Docker containerization

---

## Inspiration

Built while studying:

**Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**

by Aurélien Géron

---

## Author

**Hamza Uzzaman**

B.Tech CSE
Machine Learning Enthusiast
Focused on building production-grade ML systems

---

## License

This project is for educational and portfolio purposes.

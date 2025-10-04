YouTube Trends Analysis

A comprehensive data analysis and machine learning project exploring YouTube trending videos across multiple countries to uncover viewership patterns, category popularity, and build predictive models.

📊 Project Overview

This project analyzes YouTube trending video data from a local data warehouse, focusing on viewership patterns across different content categories in various countries. The analysis provides insights into what types of content gain traction in different regions and includes machine learning models to predict video popularity.

## 🗂️ Data Structure

The project uses a relational data model with two main components:

### Video Data (CSV)
- `video_id` (Primary Key)
- `title`
- `channel_title`
- `publish_time`
- `tags`
- `views`
- `likes`
- `dislikes`
- `comment_count`
- `trending_date`
- `category_id` (Foreign Key → Category)

### Category Data (JSON)
- `category_id` (Primary Key)
- `category_name`

## 🌍 Countries Analyzed

- United States (US)
- Russia (RU)
- Canada (CA)
- France (FR)
- India (IN)
- Japan (JP)
- Mexico (MX)
- Germany (DE)

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **JSON** - Data parsing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning models
- **Seaborn** - Enhanced visualizations

## 📈 Key Features

### 1. Data Extraction & Transformation
- Automated pipeline for processing multiple country datasets
- Integration of CSV video data with JSON category information
- Data cleaning and aggregation
- Feature engineering for ML models

### 2. Exploratory Data Analysis
- Category-wise view analysis for each country
- Sorting by total views to identify popular content types
- Comparative analysis across different regions
- Statistical analysis of engagement metrics

### 3. Machine Learning Models
- **Popularity Prediction**: Classification model to predict if a video will be highly popular
- **Feature Importance**: Analysis of which factors most influence video popularity
- **Model Evaluation**: Performance metrics and validation

### 4. Visual Representation
- Data visualization capabilities using Matplotlib and Seaborn
- Clear presentation of trending categories
- Model performance visualizations
- Feature importance charts

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Required libraries: pandas, matplotlib, seaborn, scikit-learn, numpy

### Installation
1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn numpy
   ```

### Usage
The main analysis can be run by executing the Jupyter notebook, which will:
- Process data for all specified countries
- Generate aggregated view statistics by category
- Create visual representations of the data
- Train and evaluate machine learning models
- Provide insights into video popularity factors

## 🤖 Machine Learning Component

### Model Architecture
- **Algorithm**: Various classification algorithms (Random Forest, Logistic Regression, etc.)
- **Target Variable**: Video popularity (binary classification - high vs low popularity)
- **Features Used**:
  - Category information
  - Engagement metrics (likes, dislikes, comment count)
  - Temporal features
  - Channel information

### Model Performance
- Accuracy metrics
- Precision and recall scores
- Feature importance analysis
- Cross-validation results

### Key Insights from ML
- Identification of the most important factors for video popularity
- Category-specific popularity patterns
- Engagement metric correlations

## 📊 Sample Output

The analysis provides:
- Aggregated view counts by category, sorted in ascending order
- Machine learning model performance metrics
- Feature importance rankings
- Visualizations of category popularity across regions

## 🔍 Insights

The project reveals interesting patterns in content consumption across different countries, helping content creators and marketers understand:
- Regional content preferences
- Category popularity trends
- Cross-cultural viewing patterns
- Predictive factors for video success
- Optimal posting strategies based on category and region

## 📊 Visualizations

1. **Category View Distribution** - Bar charts showing view distribution across categories
2. **Model Performance Metrics** - Confusion matrices, ROC curves
3. **Feature Importance** - Horizontal bar charts showing most influential features
4. **Geographic Comparisons** - Side-by-side category popularity across countries

## 📁 Project Structure

```
youtube_trends_analysis/
├── data/
│   ├── {country}_videos.csv
│   └── {country}_category_id.json
├── models/
│   └── trained_models/ (generated)
├── notebooks/
│   └── Youtube_trends_analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── visualization.py
│   └── ml_models.py
├── results/
│   ├── visualizations/
│   └── model_metrics/
└── README.md
```

## 🎯 Business Applications

- **Content Strategy**: Help creators understand what content performs best in different regions
- **Marketing Insights**: Inform advertising strategies based on category popularity
- **Platform Optimization**: Guide YouTube's recommendation algorithms
- **Trend Prediction**: Early identification of emerging content trends

## 👨‍💻 Author

**Abdullah Mahdy**

## 📄 License

This project is for educational and analytical purposes.

---

*Note: This project works with local YouTube trends data and requires the appropriate dataset files to be placed in the specified directory structure. The ML models are trained on historical data to predict future video performance patterns.*
```

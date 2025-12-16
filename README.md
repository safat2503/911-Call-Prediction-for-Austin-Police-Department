# 911 Call Prediction - Machine Learning Project (Group 4)

This project predicts 911 call volumes using machine learning models, comparing predictions at two levels of geographic granularity: **Sector-based** and **GeoID-based** aggregations.

## Project Overview

The project analyzes 911 call data to predict call volumes across different time aggregations (daily, weekly, monthly) and geographic regions. It compares two approaches:
- **Sector-based predictions**: 11 geographic sectors
- **GeoID-based predictions**: 676 geographic IDs (more granular)

## Project Structure

```
ML_Final_Project/
├── Data_Preprocessing.ipynb              # Initial data loading and cleaning
├── Feature_Engineering.ipynb            # Feature engineering for Sector models
├── Feature_Engineering_GeoID.ipynb      # Feature engineering for GeoID models
├── Enhanced_Model_Training.ipynb          # Model training for Sector models
├── Enhanced_Model_Training_GeoID.ipynb   # Model training for GeoID models
├── Error_Analysis.ipynb                  # Error analysis for Sector models
├── Error_Analysis_GeoID.ipynb            # Error analysis for GeoID models
├── export.csv                            # Raw input data
├── README.md                             # This file
├── requirements.txt                      # Python package dependencies
│
├── sector_daily_enhanced.csv             # Enhanced daily features for Sector (created by Feature_Engineering.ipynb)
├── sector_weekly_enhanced.csv            # Enhanced weekly features for Sector (created by Feature_Engineering.ipynb)
├── sector_monthly_enhanced.csv           # Enhanced monthly features for Sector (created by Feature_Engineering.ipynb)
├── geoid_daily_enhanced.csv              # Enhanced daily features for GeoID (created by Feature_Engineering_GeoID.ipynb)
├── geoid_weekly_enhanced.csv             # Enhanced weekly features for GeoID (created by Feature_Engineering_GeoID.ipynb)
├── geoid_monthly_enhanced.csv            # Enhanced monthly features for GeoID (created by Feature_Engineering_GeoID.ipynb)
│
├── dataset_visualization/                # Data exploration visualizations (created by Data_Preprocessing.ipynb)
├── Sector_Model_Comparison_Results/      # Sector model outputs and saved models(created by Enhanced_Model_Training.ipynb)
├── Geoid_Model_Comparison_Results/       # GeoID model outputs and saved models (created by Enhanced_Model_Training_GeoID.ipynb)
├── Sector_error_analysis/                # Sector error analysis results (created by Error_Analysis.ipynb)
└── GeoID_error_analysis/                 # GeoID error analysis results (created by Error_Analysis_GeoID.ipynb)
└── feature_engineering_comparison/       # Sector & GeoID target class distribution (created by Feature-Engineering.ipynb & Feature_Engineering_GeoID.ipynb)
```

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository** (if using Git):
   ```bash
   git clone <repository-url>
   cd ML_Final_Project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all required packages with compatible versions. Alternatively, you can install packages individually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
   ```

## Data Requirements

The project requires an input CSV file named `export.csv` in the root directory. The file can be download from the onedrive link: https://txst-my.sharepoint.com/:x:/g/personal/bet68_txstate_edu/IQARu6Q3PpolT46WzwcDzDoJAf6YtV6i025JiHaIFcBNrQI?e=3aeSbh&download=1


## How to Run the Code

### Execution Order

The notebooks should be executed in the following order:

#### 1. Data Preprocessing
```bash
jupyter notebook Data_Preprocessing.ipynb
```
- Loads and cleans the raw data (`export.csv`)
- Extracts date components
- Creates visualizations for data exploration
- **Output**: Cleaned dataset ready for feature engineering

#### 2. Feature Engineering

**For Sector-based models:**
```bash
jupyter notebook Feature_Engineering.ipynb
```
- Creates enhanced features (priority percentages, mental health flags, lag features, etc.)
- Aggregates data by Sector at daily, weekly, and monthly levels
- **Output**: 
  - `sector_daily_enhanced.csv`
  - `sector_weekly_enhanced.csv`
  - `sector_monthly_enhanced.csv`

**For GeoID-based models:**
```bash
jupyter notebook Feature_Engineering_GeoID.ipynb
```
- Creates enhanced features for GeoID-based predictions
- Aggregates data by GeoID at daily, weekly, and monthly levels
- **Output**:
  - `geoid_daily_enhanced.csv`
  - `geoid_weekly_enhanced.csv`
  - `geoid_monthly_enhanced.csv`

#### 3. Model Training

**For Sector-based models:**
```bash
jupyter notebook Enhanced_Model_Training.ipynb
```
- Trains multiple models (Linear Regression, Lasso, Random Forest, XGBoost)
- Compares model performance across daily, weekly, and monthly aggregations
- Saves trained models and generates comparison visualizations
- **Output**: 
  - Trained models in `Sector_Model_Comparison_Results/models/`
  - Performance metrics and visualizations in `Sector_Model_Comparison_Results/`

**For GeoID-based models:**
```bash
jupyter notebook Enhanced_Model_Training_GeoID.ipynb
```
- Trains models for GeoID-based predictions
- **Output**:
  - Trained models in `Geoid_Model_Comparison_Results/models/`
  - Performance metrics and visualizations in `Geoid_Model_Comparison_Results/`

#### 4. Error Analysis

**For Sector-based models:**
```bash
jupyter notebook Error_Analysis.ipynb
```
- Analyzes prediction errors by sector
- Identifies peak vs normal demand patterns
- Examines worst failure cases
- **Output**: Error analysis results in `Sector_error_analysis/`

**For GeoID-based models:**
```bash
jupyter notebook Error_Analysis_GeoID.ipynb
```
- Analyzes prediction errors by GeoID
- **Output**: Error analysis results in `GeoID_error_analysis/`

### Running All Notebooks

To run all notebooks in sequence, execute them in the order listed above. Each notebook depends on outputs from previous notebooks.

**Note**: Make sure each notebook completes successfully before moving to the next one, as later notebooks depend on CSV files and model files generated by earlier notebooks.

## Model Details

### Models Trained

1. **Linear Regression** - Baseline linear model
2. **Lasso Regression** - Regularized linear model with feature selection
3. **Random Forest** - Ensemble tree-based model
4. **XGBoost** - Gradient boosting model

### Features Used

- Temporal features: Month, Year, Day of Year, Week
- Priority features: Percentage of Priority 1-4 calls
- Mental Health: Percentage of mental health incidents
- Category features: Top 5 incident category percentages
- Lag features: Previous day/week/month values
- Temporal indicators: Weekend, holiday, peak day flags
- Rolling statistics: 7-day, 30-day rolling means and standard deviations

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

## Output Files

### Generated Datasets
- `sector_daily_enhanced.csv`, `sector_weekly_enhanced.csv`, `sector_monthly_enhanced.csv`
- `geoid_daily_enhanced.csv`, `geoid_weekly_enhanced.csv`, `geoid_monthly_enhanced.csv`

### Saved Models
- Models saved as `.joblib` files in respective results directories
- Scalers saved for preprocessing new data

### Visualizations
- Model comparison charts
- Feature importance plots
- Actual vs Predicted scatter plots
- Error analysis visualizations

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `export.csv` is in the root directory
2. **ModuleNotFoundError**: Install missing packages using `pip install <package-name>`
3. **Memory errors**: Reduce data size or use a machine with more RAM
4. **Model loading errors**: Ensure previous notebooks have been executed to generate model files

### Dependencies

If you encounter version conflicts, the project was developed with:
- Python 3.9.6
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0

## Notes

- The notebooks use a temporal train-test split (80/20) with `random_state=42` for reproducibility
- Models are trained separately for Sector and GeoID aggregations
- All visualizations are saved automatically in respective output directories
- **Model files (`.joblib`) are not included in the repository** due to their large size. Run the model training notebooks to generate them locally.

## Contact
uzd14@txstate.edu

 U p d a t e  
 
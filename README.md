# Exploratory Data Analysis - Financial Loans

## Table of Contents
1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [File Structure](#file-structure)
5. [License](#license)

## Project Description
This project conducts **Exploratory Data Analysis (EDA)** on a financial loans dataset. The primary goal is to gain insights into the portfolio of a loan company by cleaning, transforming, and analysing the data.

### Objectives:
- Load and inspect the dataset
- Handle missing values and data inconsistencies 
- Convert data types appropriately (e.g., categorical encoding, datetime conversion)
- Perform descriptive statistical analysis
- Visualise key trends and relationships

Through this project, I enhanced my understanding of **data preprocessing**, **data handling**, and **basic financial data analysis**.

## Installation Instructions
To run this project, ensure you have Python installed along with the required dependencies.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the EDA Jupyter Notebook:
   ```bash
   jupyter notebook EDA.ipynb
   ```
4. Run the EDA Analysis Jupyter Notebook.
   ```bash
   jupyter notebook EDA_analysis.ipynb
   ```

## Usage Instructions
1. Open the EDA Jupyter Notebook (`EDA.ipynb`).
2. Run the cells sequentially to:
   - Load the loan dataset
   - Perform data type conversions and cleaning
   - Explore statistical summaries
   - Handle skewed data, remove outliers and drop overly correlated columns
3. Open the EDA Analysis Jupyter Notebook (`EDA_analysis.ipynb`).
4. Run the cells sequentially to:
   - Load the clean data
   - Assess the current state of loans
   - Calculate loss
   - Investigate predictors of loss

## File Structure
```
project-directory/
│── EDA.ipynb                    # Jupyter Notebook with exploratory data analysis
│── EDA_analysis.ipynb           # Jupyter Notebook with further data analysis
│── loans_data_raw.csv           # Raw dataset
│── loan_dataset_schema.ipynb    # Definitions of each column in dataset
│── requirements.txt             # List of dependencies
│── README.md                    # Project documentation
└── loan_data_analysis.py        # Helper functions for data processing
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

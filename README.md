# Simple-Linear-Regression
# Weight-Height Prediction using Linear Regression

This project demonstrates how to predict height based on weight using a simple linear regression model in Python. It utilizes libraries like Pandas, Matplotlib, NumPy, Seaborn, and Scikit-learn.

## Dataset

The project uses the "Weight-Height Polynomial Dataset.csv" file, which contains weight and height data.

## Libraries Used

* Pandas: For data manipulation and analysis.
* Matplotlib: For creating visualizations.
* NumPy: For numerical computations.
* Seaborn: For enhanced visualizations.
* Scikit-learn: For building and evaluating the linear regression model.

## Methodology

1. **Data Loading and Exploration:** The dataset is loaded using Pandas, and basic exploratory data analysis is performed.
2. **Data Visualization:** Scatter plots and pair plots are used to visualize the relationship between weight and height.
3. **Data Preprocessing:** The data is split into training and testing sets using `train_test_split`. The weight feature is standardized using `StandardScaler`.
4. **Model Building:** A linear regression model is created and trained using the training data.
5. **Model Evaluation:** The model's performance is evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.
6. **Prediction:** The trained model is used to predict height for new weight values.

## Usage

1. **Clone the repository:** `git clone <repository_url>`
2. **Install dependencies:** `pip install pandas matplotlib numpy seaborn scikit-learn`
3. **Run the Jupyter Notebook:** Open and run the notebook to see the code and results.

## Results

The model achieves an R-squared score of [insert R-squared score here], indicating a [good/moderate/poor] fit to the data.


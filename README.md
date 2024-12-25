# Titanic Survival Prediction

## Objectives
Develop a machine learning model to predict whether a passenger survived the Titanic disaster, using features like age, gender, ticket class, and fare.

## Approach
1. **Data Preprocessing**:
   - Handled missing values in `Age`, `Embarked`, and `Cabin`.
   - Created new features: `Title`, `FamilySize`, `IsAlone`, `AgeBin`, and `FareBin`.
   - Encoded categorical variables.

2. **Feature Selection**:
   - Selected relevant features like `Pclass`, `Sex`, `AgeBin`, `FareBin`, and `IsAlone`.

3. **Model Building**:
   - Trained and evaluated multiple models: Logistic Regression, Random Forest, etc.
   - Tuned hyperparameters using GridSearchCV.

4. **Evaluation**:
   - Evaluated models using accuracy, precision, recall, F1-score, and a confusion matrix.

## Challenges
- Handling missing data for `Age` and `Cabin`.
- Balancing the dataset (if applicable).
- Selecting optimal model hyperparameters.

## Results
- Best model: Random Forest Classifier.
- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- F1-Score: XX%

## Repository Structure

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'StudentPerformance_cleaned.csv'

try:
    df = pd.read_csv(file_path)

    # Create binary target: PassedReading (1 = MathScore >= 50)
    df['PassedReading'] = (df['MathScore'] >= 50).astype(int)
    print("Target variable 'PassedReading' added.\n")
    print(df.head())

    # Select features
    feature_columns = ['ReadingScore', 'WritingScore', 'Gender_male',
                       'LunchType_standard', 'TestPrepCourse_none', 'TestPrepCourse_not completed']

    X = df[feature_columns]
    y = df['PassedReading']

    print("\nFeatures (X) head:")
    print(X.head())
    print("\nTarget (y) head:")
    print(y.head())

    # Split data into training and test sets
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"\nTraining and testing data prepared. Train shape: {X_train_r.shape}")

    # Train Gaussian Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train_r, y_train_r)
    print("\nGaussian Naive Bayes model trained.")

    # Make predictions with Naive Bayes
    y_pred_nb = nb_model.predict(X_test_r)
    print("Predictions made with Naive Bayes.")

    # Evaluate Naive Bayes model
    accuracy_nb = accuracy_score(y_test_r, y_pred_nb)
    conf_matrix_nb = confusion_matrix(y_test_r, y_pred_nb)
    class_report_nb = classification_report(y_test_r, y_pred_nb)

    print(f"\nNaive Bayes Accuracy: {accuracy_nb:.4f}")
    print("\nNaive Bayes Confusion Matrix:")
    print(conf_matrix_nb)
    print("\nNaive Bayes Classification Report:")
    print(class_report_nb)

    # Confusion matrix heatmap
    sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Naive Bayes Confusion Matrix (PassedReading)')
    plt.show()

    # Train Decision Tree model (max_depth=4)
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train_r, y_train_r)
    y_pred_dt = dt_model.predict(X_test_r)

    accuracy_dt = accuracy_score(y_test_r, y_pred_dt)
    class_report_dt = classification_report(y_test_r, y_pred_dt)

    print(f"\nDecision Tree Accuracy (max_depth=4): {accuracy_dt:.4f}")
    print("\nDecision Tree Classification Report:")
    print(class_report_dt)

    # Compare Precision & Recall for class '1' (Passed)
    print("\nComparison for Class '1' (Passed):")

    precision_nb = precision_score(y_test_r, y_pred_nb)
    recall_nb = recall_score(y_test_r, y_pred_nb)
    precision_dt = precision_score(y_test_r, y_pred_dt)
    recall_dt = recall_score(y_test_r, y_pred_dt)

    print(f"Naive Bayes - Precision: {precision_nb:.4f}, Recall: {recall_nb:.4f}")
    print(f"Decision Tree - Precision: {precision_dt:.4f}, Recall: {recall_dt:.4f}")

    print("\nComment:")
    if precision_nb > precision_dt:
        print("- Naive Bayes gives better precision for 'Passed'.")
    else:
        print("- Decision Tree gives better precision for 'Passed'.")

    if recall_nb > recall_dt:
        print("- Naive Bayes gives better recall for 'Passed'.")
    else:
        print("- Decision Tree gives better recall for 'Passed'.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")

import pandas as pd


def preprocess_csv(src: str, dest: str):
    """
    Preprocess data from source CSV file and save the processed data to a destination CSV file.

    Args:
        src (str): Path to the source CSV file.
        dest (str): Path to the destination CSV file where the preprocessed data will be saved.

    Returns:
        None
    """
    data = pd.read_csv(src)

    # Map general health to numerical values
    general_health_map = {
        "Poor": 0.0,
        "Fair": 1.0,
        "Good": 2.0,
        "Very Good": 3.0,
        "Excellent": 4.0,
    }
    data["General_Health"] = data["General_Health"].map(general_health_map)

    # Map checkup to numerical values
    checkup_map = {
        "Never": 0.0,
        "5 or more years ago": 1.0,
        "Within the past 5 years": 2.0,
        "Within the past 2 years": 3.0,
        "Within the past year": 4.0,
    }
    data["Checkup"] = data["Checkup"].map(checkup_map)

    # Map exercise to numerical values
    exercise_map = {"No": 0.0, "Yes": 1.0}
    data["Exercise"] = data["Exercise"].map(exercise_map)

    # Map skin_cancer to numerical values
    skin_cancer_map = {"No": 0.0, "Yes": 1.0}
    data["Skin_Cancer"] = data["Skin_Cancer"].map(skin_cancer_map)

    # Map other_cancer to numerical values
    other_cancer_map = {"No": 0.0, "Yes": 1.0}
    data["Other_Cancer"] = data["Other_Cancer"].map(other_cancer_map)

    # Map depression to numerical values
    depression_map = {"No": 0.0, "Yes": 1.0}
    data["Depression"] = data["Depression"].map(depression_map)

    # Map diabetes to numerical values
    diabetes_map = {
        "No": 0.0,
        "No, pre-diabetes or borderline diabetes": 0.0,
        "Yes, but female told only during pregnancy": 1.0,
        "Yes": 1.0,
    }
    data["Diabetes"] = data["Diabetes"].map(diabetes_map)

    # Map arthritis to numerical values
    arthritis_map = {"No": 0.0, "Yes": 1.0}
    data["Arthritis"] = data["Arthritis"].map(arthritis_map)

    # Map gender to numerical values
    gender_map = {"Male": 0.0, "Female": 1.0}
    data["Sex"] = data["Sex"].map(gender_map)

    # Map age_category to numerical values
    age_category_map = {
        "18-24": 0.0,
        "25-29": 1.0,
        "30-34": 2.0,
        "35-39": 3.0,
        "40-44": 4.0,
        "45-49": 5.0,
        "50-54": 6.0,
        "55-59": 7.0,
        "60-64": 8.0,
        "65-69": 9.0,
        "70-74": 10.0,
        "75-79": 11.0,
        "80+": 12.0,
    }
    data["Age_Category"] = data["Age_Category"].map(age_category_map)
    # Normalizing the age_category
    age_max = max(data["Age_Category"])
    data["Age_Category"] = data["Age_Category"] / age_max

    # Map heart_disease to numerical values
    heart_disease_map = {"No": 0.0, "Yes": 1.0}
    data["Heart_Disease"] = data["Heart_Disease"].map(heart_disease_map)

    # Map smoking_history to numerical values
    smoking_history_map = {"No": 0.0, "Yes": 1.0}
    data["Smoking_History"] = data["Smoking_History"].map(smoking_history_map)

    data["Alcohol_Consumption"] = pd.to_numeric(
        data["Alcohol_Consumption"], errors="coerce"
    )
    data["Fruit_Consumption"] = pd.to_numeric(
        data["Fruit_Consumption"], errors="coerce"
    )
    data["Green_Vegetables_Consumption"] = pd.to_numeric(
        data["Green_Vegetables_Consumption"], errors="coerce"
    )
    data["FriedPotato_Consumption"] = pd.to_numeric(
        data["FriedPotato_Consumption"], errors="coerce"
    )
    data["Height_(cm)"] = pd.to_numeric(data["Height_(cm)"], errors="coerce")
    data["Weight_(kg)"] = pd.to_numeric(data["Weight_(kg)"], errors="coerce")
    data["BMI"] = pd.to_numeric(data["BMI"], errors="coerce")

    # Max of each column
    alchohol_max = max(data["Alcohol_Consumption"])
    fruit_max = max(data["Fruit_Consumption"])
    green_veg_max = max(data["Green_Vegetables_Consumption"])
    fried_potato_max = max(data["FriedPotato_Consumption"])
    height_max = max(data["Height_(cm)"])
    weight_max = max(data["Weight_(kg)"])
    bmi_max = max(data["BMI"])

    # Normalize the columns using the max
    data["Alcohol_Consumption"] = data["Alcohol_Consumption"] / alchohol_max
    data["Fruit_Consumption"] = data["Fruit_Consumption"] / fruit_max
    data["Green_Vegetables_Consumption"] = (
        data["Green_Vegetables_Consumption"] / green_veg_max
    )
    data["FriedPotato_Consumption"] = data["FriedPotato_Consumption"] / fried_potato_max
    data["Height_(cm)"] = data["Height_(cm)"] / height_max
    data["Weight_(kg)"] = data["Weight_(kg)"] / weight_max
    data["BMI"] = data["BMI"] / bmi_max

    # Select specific columns to keep
    columns_to_keep = [
        "General_Health",
        "Checkup",
        "Exercise",
        "Skin_Cancer",
        "Other_Cancer",
        "Depression",
        "Diabetes",
        "Arthritis",
        "Sex",
        "Age_Category",
        "Height_(cm)",
        "Weight_(kg)",
        "BMI",
        "Smoking_History",
        "Alcohol_Consumption",
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "FriedPotato_Consumption",
        "Heart_Disease",
    ]
    data = data[columns_to_keep]

    # Convert non-object columns to float
    for column in data.columns:
        if data[column].dtype != object:
            data[column] = data[column].astype(float)

    data.to_csv(dest, index=False)

    return


def balance_dataset(data):

    # Balance data due to extreme class imbalance (92% negative, 8% positive)
    data_majority = data[data.iloc[:, -1] == 0]
    data_minority = data[data.iloc[:, -1] == 1]

    # Oversample minority class
    minority_size = data_minority.shape[0]
    majority_size = data_majority.shape[0]

    # Repeat the minority class rows until it matches the size of the majority class
    oversampled_minority = data_minority.loc[
        data_minority.index.repeat((majority_size // minority_size) + 1)
    ].reset_index(drop=True)
    oversampled_minority = oversampled_minority.iloc[:majority_size, :]

    # Combine majority class with oversampled minority class
    data_balanced = pd.concat([data_majority, oversampled_minority])

    # Shuffle the data
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data_balanced.iloc[:, :-1].values
    Y = data_balanced.iloc[:, -1].values

    return X, Y

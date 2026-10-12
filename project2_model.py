from pathlib import Path
from zipfile import ZipFile
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import classification_report, precision_score, recall_score, precision_recall_curve
from xgboost import XGBClassifier
import warnings

# Relative paths
base_dir = Path(__file__).resolve().parent

#Path of project files
raw_zip_path = base_dir / "project2_raw_data.zip"
mapper_path = base_dir / "feature_name_to_number_mapping.csv"

#Creating directory for processed files and their paths
out_dir = base_dir / "data" / "processed"
flat_file_path = out_dir / "modelling_data.csv"
feature_dict_path = out_dir / "feature_dictionary.csv"
performance_path = out_dir / "model_performance.txt"
feature_importance_path = out_dir / "feature_importance.csv"
feature_importance_plot_path = out_dir / "feature_importance_top20.png"
model_package_path = out_dir / "packaged_model.pkl"
baseline_performance_path = out_dir / "model_performance_baseline.txt"
scale_pos_weight_performance_path = out_dir / "model_performance_scale_pos_weight.txt"
threshold_search_path = out_dir / "threshold_search_results.csv"
weighted_threshold_search_path = out_dir / "threshold_search_results_scale_pos_weight.csv"

# Helper functions
#This reads the raw month files and converts them into a flat data frame
def read_raw_data(raw_zip_path, mapper):
    feature_map = dict(zip(mapper["feature_number"], mapper["feature_name_unique"]))
    feature_cols = mapper["feature_name_unique"].tolist()

    rows = []

    with ZipFile(raw_zip_path, "r") as zf:
        txt_files = sorted([name for name in zf.namelist() if name.endswith(".txt")])

        for txt_file in txt_files:
            month = pd.to_datetime(Path(txt_file).stem, format="%Y-%m")

            with zf.open(txt_file) as f:
                for raw_line in f:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    parts = line.split()

                    row = {"risk_score": float(parts[0]), "file_month": month}

                    for token in parts[1:]:
                        if ":" in token:
                            feat_num, feat_val = token.split(":", 1)
                            feat_num = int(feat_num)
                            feat_val = float(feat_val)

                            if feat_num in feature_map:
                                row[feature_map[feat_num]] = feat_val

                    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=["risk_score", "file_month"] + feature_cols)
    # Not addressing raw feature values at this stage as XGBoost will handle it
    return df

#This makes the feature names lower case and consistent
def clean_feature_name(value):
    value = str(value).lower().strip().replace("/", "_")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", value)).strip("_")

#This creates the features
def add_engineered_features(df, mapper):
    df = df.copy()

    base_feature_cols = mapper["feature_name_unique"].tolist()
    # The engineered summary features skips NaN, the original raw features stay NaN for XGBoost to handle
    engineered_features = {}

    engineered_features["total_feature_sum"] = df[base_feature_cols].sum(axis=1)
    engineered_features["active_feature_count"] = (df[base_feature_cols] > 0).sum(axis=1)

    type_sum_cols = []

    for feature_type in mapper["feature_type"].unique():
        cols = mapper.loc[mapper["feature_type"] == feature_type, "feature_name_unique"].tolist()

        safe_name = clean_feature_name(feature_type)
        col_name = f"type_{safe_name}_sum"

        engineered_features[col_name] = df[cols].sum(axis=1)
        type_sum_cols.append(col_name)

    subtype_sum_cols = []

    for feature_subtype in mapper["feature_subtype"].unique():
        cols = mapper.loc[mapper["feature_subtype"] == feature_subtype, "feature_name_unique"].tolist()

        safe_name = clean_feature_name(feature_subtype)
        col_name = f"subtype_{safe_name}_sum"

        engineered_features[col_name] = df[cols].sum(axis=1)
        subtype_sum_cols.append(col_name)

    engineered_df = pd.DataFrame(engineered_features, index=df.index)

    engineered_df["dominant_feature_type_share"] = (engineered_df[type_sum_cols].max(axis=1)
    / engineered_df["total_feature_sum"].replace(0, 1))

    engineered_df["dominant_feature_subtype_share"] = (engineered_df[subtype_sum_cols].max(axis=1)
    / engineered_df["total_feature_sum"].replace(0, 1))

    df = pd.concat([df, engineered_df], axis=1)

    return df

#Creates risk_level and adds features
def make_features(df, mapper):
    df = df.copy()
    # From the spec, the high risk scores are >= 0.30, those below are low risk
    df["risk_level"] = (df["risk_score"] >= 0.30).astype(int)
    df = add_engineered_features(df, mapper)

    return df

# This function is used for the packaged model.
def prepare_new_data(raw_rows, mapper, feature_columns):
    feature_map = dict(zip(mapper["feature_number"], mapper["feature_name_unique"]))
    base_feature_cols = mapper["feature_name_unique"].tolist()

    rows = []
    bad_token_count = 0
    unknown_feature_count = 0

    for row_index, line in enumerate(raw_rows, start=1):
        line = str(line).strip()

        if not line:
            raise ValueError(f"Input row {row_index} is empty.")

        row = {}
        valid_feature_count = 0

        for token in line.split():
            left, sep, right = token.partition(":")

            if sep != ":":
                bad_token_count+= 1
                continue

            try:
                feat_num = int(left)
                feat_val = float(right)
            except ValueError:
                bad_token_count+= 1
                continue

            if feat_num in feature_map:
                row[feature_map[feat_num]] = feat_val
                valid_feature_count += 1
            else:
                unknown_feature_count+= 1

        if valid_feature_count == 0:
            raise ValueError(
                f"Input row {row_index} does not contain any valid mapped feature tokens.")
        rows.append(row)

#Makes a warning if any input tokens cant be used for prediction
    if bad_token_count > 0:
        warnings.warn(f"Dropped {bad_token_count}, incorrect format token(s).",
        UserWarning)

    if unknown_feature_count > 0:
        warnings.warn(f"Dropped {unknown_feature_count}, feature number(s) not found in mapper.",
        UserWarning)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=base_feature_cols)

    df = add_engineered_features(df, mapper)

    df = df.reindex(columns=feature_columns)

    return df

#Evaluates the model, on a split using the chosen classification threshold
def evaluate_model(model, x, y, split_name, threshold):

    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= threshold).astype(int)

    report = classification_report(y, preds, zero_division=0)

    print()
    print(split_name)
    print(report)

    return f"\n{split_name}\n{report}\n"

#The low risk and high risk counts for the split
def split_class_balance(split_df, split_name):
    high_risk_rate = split_df["risk_level"].mean()
    class_counts = split_df["risk_level"].value_counts().sort_index()

    text = (f"\n{split_name} class balance\n"
        f"Low risk count: {class_counts.get(0, 0)}\n"
        f"High risk count: {class_counts.get(1, 0)}\n"
        f"High risk rate: {high_risk_rate:.4f}\n")

    print(text)
    return text

#This chooses the threshold that has a precision and recall above 90%
def choose_validation_threshold(y_true, probabilities, min_precision=0.90, min_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    threshold_df = pd.DataFrame({"threshold": thresholds, "high_risk_precision": precision[:-1],
    "high_risk_recall": recall[:-1]})

    threshold_df["high_risk_f1"] = (2 * threshold_df["high_risk_precision"]* threshold_df["high_risk_recall"]
    / (threshold_df["high_risk_precision"]+ threshold_df["high_risk_recall"])).fillna(0)

    valid_thresholds = threshold_df[(threshold_df["high_risk_precision"] >= min_precision)
    & (threshold_df["high_risk_recall"] >= min_recall)].copy()

    if valid_thresholds.empty:
        raise ValueError("No validation threshold meets both 90% high-risk precision "
            "and 90% high-risk recall.")

    best_row = valid_thresholds.sort_values(by=["high_risk_f1", "high_risk_precision", "high_risk_recall"],
    ascending=False).iloc[0]

    return float(best_row["threshold"]), threshold_df, best_row

#Main function, uses helper functions
def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    mapper = pd.read_csv(mapper_path)

    print(" ")
    print("rows:", len(mapper))
    print("columns:", list(mapper.columns))

    print(" ")
    print("Missing Mapper Values")
    print(mapper.isna().sum())

    mapper["feature_number"] = mapper["feature_number"].astype(int)
    mapper["feature_name"] = mapper["feature_name"].astype(str)
    mapper["feature_type"] = mapper["feature_type"].fillna("unknown")
    mapper["feature_subtype"] = mapper["feature_subtype"].fillna("N/A")

    mapper.loc[mapper["feature_subtype"].astype(str).str.strip() == "", "feature_subtype"] = "N/A"

    print(" ")
    if mapper["feature_subtype"].isna().any():
        raise ValueError("feature_subtype still has missing values.")
    else:
        print("feature_subtype missing values set to the string N/A")

    mapper = mapper.sort_values("feature_number").reset_index(drop=True)

    mapper["feature_name_clean"] = mapper["feature_name"].apply(clean_feature_name)

    mapper["feature_name_unique"] = [
        f"f{num:03d}_{name}"
        for num, name in zip(mapper["feature_number"], mapper["feature_name_clean"])]

    print(" ")
    print("Checking for missing mapper values again")
    print(mapper.isna().sum())

    print(" ")
    duplicate_featurenumbers = mapper[mapper.duplicated(subset=["feature_number"], keep=False)] \
        .sort_values("feature_number")

    if len(duplicate_featurenumbers) == 0:
        print("No duplicate feature numbers")
    else:
        print(duplicate_featurenumbers)
        raise ValueError("Duplicate feature numbers found in mapper.")

    print(" ")
    print("Checking for duplicate feature names in mapper")
    duplicate_featurenames = mapper[mapper.duplicated(subset=["feature_name"], keep=False)] \
        .sort_values(["feature_name", "feature_number"])
    if len(duplicate_featurenames) == 0:
        print("none")
    else:
        print(duplicate_featurenames[["feature_name", "feature_number", "feature_type", "feature_subtype"]])

    print(" ")
    print("Example of final feature names")
    print("This shows that duplicate feature names are kept separate using feature numbers")
    print(mapper[["feature_number", "feature_name", "feature_name_unique"]].head(10))

    mapper.to_csv(feature_dict_path, index=False)

    df = read_raw_data(raw_zip_path, mapper)

    feature_cols = mapper["feature_name_unique"].tolist()

    print(" ")

    feature_cols_missing = [col for col in feature_cols if col not in df.columns]

    if feature_cols_missing:
        raise ValueError(f"These feature_name_unique columns are missing: {feature_cols_missing}")
    else:
        print("All feature_name_unique columns exist")

    missing_feature_count = df[feature_cols].isna().sum().sum()
    print(f"Missing raw feature values left as NaN: {missing_feature_count}")
    print("Missing raw feature values will be handled by XGBoost")

    df = make_features(df, mapper)

    non_model_cols = ["risk_score", "risk_level", "file_month"]
    model_cols = [col for col in df.columns if col not in non_model_cols]

    bad_model_cols = [col for col in model_cols if not is_numeric_dtype(df[col])]

    if bad_model_cols:
        raise TypeError(f"The following modelling columns should be number type: {bad_model_cols}")
    elif not is_numeric_dtype(df["risk_score"]):
        raise TypeError("risk_score should be number type.")
    elif not set(df["risk_level"].dropna().unique()).issubset({0, 1}):
        raise TypeError("risk_level should only be 0 and 1.")
    elif not is_datetime64_any_dtype(df["file_month"]):
        raise TypeError(f"file_month should be datetime, got {df['file_month'].dtype}")
    else:
        print("All data types are correct for modelling")

    df.to_csv(flat_file_path, index=False)

    months = sorted(df["file_month"].unique())

    train_cut = int(len(months) * 0.6)
    valid_cut = int(len(months) * 0.8)

    train_df = df[df["file_month"].isin(months[:train_cut])].copy()
    valid_df = df[df["file_month"].isin(months[train_cut:valid_cut])].copy()
    test_df = df[df["file_month"].isin(months[valid_cut:])].copy()

    performance_text = ""

    performance_text += split_class_balance(train_df, "Train")
    performance_text += split_class_balance(valid_df, "Validation")
    performance_text += split_class_balance(test_df, "Test")

    x_train = train_df.drop(columns=["risk_level", "risk_score", "file_month"])
    y_train = train_df["risk_level"]

    x_valid = valid_df.drop(columns=["risk_level", "risk_score", "file_month"])
    y_valid = valid_df["risk_level"]

    x_test = test_df.drop(columns=["risk_level", "risk_score", "file_month"])
    y_test = test_df["risk_level"]

    print(" ")
    print("train shape:", x_train.shape)
    print("valid shape:", x_valid.shape)
    print("test shape :", x_test.shape)

    # Testing a weighted XGBoost model (scale_pos_weight) as the classes are not balanced. The scale_pos_weight is below
    # 1 as high risk is the much larger class. This weighted model is compared on validation data, I will select the
    # most appropriate model from the comparison to be packaged.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(" ")
    print("Training low-risk count:", negative_count)
    print("Training high-risk count:", positive_count)
    print("scale_pos_weight:", scale_pos_weight)

    # Baseline model, no weighting adjustment

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        tree_method="hist")

    print(" ")
    model.fit(x_train, y_train, verbose=False)
    valid_probs = model.predict_proba(x_valid)[:, 1]

    chosen_threshold, threshold_df, chosen_threshold_row = choose_validation_threshold(
    y_valid,valid_probs,min_precision=0.90,min_recall=0.90)

    threshold_df.to_csv(threshold_search_path, index=False)

    print(" ")
    print("Chosen threshold:", chosen_threshold)
    print("Chosen threshold validation metrics:")
    print(chosen_threshold_row)

    baseline_text = ""
    baseline_text += "Baseline normal XGBoost model\n"
    baseline_text += "This is the baseline model that is evaluated against the class-weighted mitigation model.\n"
    baseline_text += f"Validation-selected classification threshold: {chosen_threshold}\n"
    baseline_text += "Threshold chosen from the validation precision-recall curve.\n"
    baseline_text += "Only thresholds with at least 90% high-risk precision and recall were considered.\n"
    baseline_text += "The selected threshold maximised high-risk F1 among thresholds that met at least 90% high-risk precision and recall.\n"
    baseline_text += "\nChosen threshold metrics:\n"
    baseline_text += chosen_threshold_row.to_string()
    baseline_text += f"\n\nFull threshold search saved to: {threshold_search_path.name}\n"
    baseline_text += f"Training low-risk count: {negative_count}\n"
    baseline_text += f"Training high-risk count: {positive_count}\n"

    baseline_text += evaluate_model(model,x_train,y_train,"Train - baseline normal XGBoost",
    chosen_threshold)

    baseline_text += evaluate_model(model,x_valid,y_valid, "Validation - baseline normal XGBoost",
    chosen_threshold)

    with open(baseline_performance_path, "w") as f:
        f.write(baseline_text)

    print("saved:", baseline_performance_path.name)

    # Weighted model using XGBoost and scale_pos_weight to see if that helps the weak low risk performance

    model_scale_pos_weight = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight)

    print(" ")
    model_scale_pos_weight.fit(x_train, y_train, verbose=False)

    weighted_valid_probs = model_scale_pos_weight.predict_proba(x_valid)[:, 1]

    try:
        weighted_threshold, weighted_threshold_df, weighted_threshold_row = choose_validation_threshold(y_valid,
        weighted_valid_probs,min_precision=0.90,min_recall=0.90)

    except ValueError as error:
        weighted_threshold = chosen_threshold
        weighted_threshold_df = None
        weighted_threshold_row = None
        weighted_threshold_error = error
    else:
        weighted_threshold_error = None

    if weighted_threshold_df is not None:
        weighted_threshold_df.to_csv(weighted_threshold_search_path, index=False)

    scale_pos_weight_text = ""
    scale_pos_weight_text += "scale_pos_weight XGBoost mitigation experiment\n"
    scale_pos_weight_text += "This model was tested to address weak low-risk recall.\n"
    scale_pos_weight_text += f"scale_pos_weight: {scale_pos_weight}\n"
    scale_pos_weight_text += f"Training low-risk count: {negative_count}\n"
    scale_pos_weight_text += f"Training high-risk count: {positive_count}\n"

    if weighted_threshold_error is None:
        scale_pos_weight_text += f"Weighted model selected threshold: {weighted_threshold}\n"
        scale_pos_weight_text += "The weighted model threshold was selected from its own validation precision-recall curve.\n"
        scale_pos_weight_text += f"Weighted threshold search saved to: {weighted_threshold_search_path.name}\n"
        scale_pos_weight_text += "\nWeighted threshold metrics:\n"
        scale_pos_weight_text += weighted_threshold_row.to_string()
        scale_pos_weight_text += "\n"
    else:
        scale_pos_weight_text += (
            "\nNo threshold for the scale_pos_weight model met both "
            "90% high-risk precision and 90% high-risk recall.\n"
        )
        scale_pos_weight_text += f"Reason: {weighted_threshold_error}\n"
        scale_pos_weight_text += f"Fallback comparison threshold used: {weighted_threshold}\n"

    scale_pos_weight_text += evaluate_model(
        model_scale_pos_weight,
        x_train,
        y_train,
        "Train - scale_pos_weight XGBoost",
        weighted_threshold
    )

    scale_pos_weight_text += evaluate_model(
        model_scale_pos_weight,
        x_valid,
        y_valid,
        "Validation - scale_pos_weight XGBoost",
        weighted_threshold
    )
    with open(scale_pos_weight_performance_path, "w") as f:
        f.write(scale_pos_weight_text)

    print("saved:", scale_pos_weight_performance_path.name)

    # Final model decision:
    # The weighted model is not definitively better, it improves low risk recall and F1 slightly but also reduces low risk precision slightly
    # But the baseline model has a slightly better high risk recall and high risk F1, and high risk performance is my main target
    # Thus the baseline model was selected to be used for the packaged model
    final_model = model
    final_threshold = chosen_threshold
    final_model_name = "baseline normal XGBoost"

    print(" ")
    print("Final selected model:", final_model_name)
    print("Final selected threshold:", final_threshold)

    performance_text += "\nClass imbalance mitigation experiment\n"
    performance_text += "A baseline normal XGBoost model and a scale_pos_weight XGBoost model were both tested.\n"
    performance_text += f"Baseline results saved to: {baseline_performance_path.name}\n"
    performance_text += f"scale_pos_weight results saved to: {scale_pos_weight_performance_path.name}\n"
    performance_text += f"scale_pos_weight tested: {scale_pos_weight}\n"
    performance_text += f"Final selected model: {final_model_name}\n"
    performance_text += f"Final selected threshold: {final_threshold}\n"

    if weighted_threshold_error is None:
        performance_text += ("The scale_pos_weight model was tested as a mitigation for weak low-risk recall. "
        "It was not selected as the final model because the improvement to low-risk "
        "validation recall was small and the baseline model gave stronger high-risk "
        "validation performance. The baseline model was therefore retained as the final "
        "packaged model.\n")
    else:
        performance_text += ("The baseline model was retained because the scale_pos_weight model could not "
        "find a validation threshold that met both 90% high-risk precision and "
        "90% high-risk recall.\n" )

    performance_text += "\nThreshold selection\n"
    performance_text += "Each model's classification threshold was selected from its own validation precision-recall curve.\n"
    performance_text += "Only thresholds with at least 90% high-risk precision and recall were considered.\n"
    performance_text += "The selected threshold maximised high-risk F1 among thresholds that met the project requirement.\n"
    performance_text += f"Baseline selected threshold: {chosen_threshold}\n"

    if weighted_threshold_error is None:
        performance_text += f"scale_pos_weight selected threshold: {weighted_threshold}\n"
        performance_text += f"Weighted threshold search saved to: {weighted_threshold_search_path.name}\n"
    else:
        performance_text += "scale_pos_weight model did not find a valid threshold meeting both requirements.\n"

    performance_text += f"Final selected threshold: {final_threshold}\n"

    #Evaluation of the final model on train, validation and test data

    performance_text += evaluate_model(final_model,x_train,y_train,f"Train - final selected model ({final_model_name})",
    final_threshold)

    performance_text += evaluate_model(final_model,x_valid,y_valid,f"Validation - final selected model ({final_model_name})",
    final_threshold)

    valid_probs = final_model.predict_proba(x_valid)[:, 1]
    valid_preds = (valid_probs >= final_threshold).astype(int)
    valid_precision = precision_score(y_valid, valid_preds, zero_division=0)
    valid_recall = recall_score(y_valid, valid_preds, zero_division=0)

    print(" ")
    print("Validation requirement check - high risk class")
    print("Precision:", valid_precision)
    print("Recall:", valid_recall)
    print("Final selected model:", final_model_name)
    print("Chosen threshold for classification:", final_threshold)

    performance_text += "\nValidation requirement check - high risk class\n"
    performance_text += f"Precision: {valid_precision}\n"
    performance_text += f"Recall: {valid_recall}\n"
    performance_text += f"Final selected model: {final_model_name}\n"
    performance_text += f"Chosen threshold for classification: {final_threshold}\n"

    if valid_precision < 0.90:
        raise ValueError("Validation precision is below 90%")
    elif valid_recall < 0.90:
        raise ValueError("Validation recall is below 90%")
    else:
        print("Validation precision and recall are both at least 90%")
        performance_text += "Validation precision and recall are both at least 90%\n"

    performance_text += evaluate_model(final_model,x_test,y_test,f"Test with final selected model ({final_model_name})",
    final_threshold)

    with open(performance_path, "w") as f:
        f.write(performance_text)


    print("saved:", performance_path.name)
    booster = final_model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")

    importance_df = pd.DataFrame({"feature": x_train.columns,
    "importance_gain": [gain_scores.get(col, 0.0) for col in x_train.columns]
    }).sort_values("importance_gain", ascending=False)

    importance_df.to_csv(feature_importance_path, index=False)

    top20 = importance_df.head(20).sort_values("importance_gain", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature"], top20["importance_gain"])
    plt.xlabel("Importance (gain)")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(feature_importance_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    package = {"model": final_model,"threshold": final_threshold,"feature_columns": x_train.columns.tolist(),
    "mapper": mapper,"model_name": final_model_name}

    with open(model_package_path, "wb") as f:
        pickle.dump(package, f)

    print("saved:", flat_file_path.name)
    print("saved:", feature_dict_path.name)
    print("saved:", feature_importance_plot_path.name)
    print("saved:", model_package_path.name)


if __name__ == "__main__":
    main()

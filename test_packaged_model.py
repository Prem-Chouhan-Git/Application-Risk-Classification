from pathlib import Path
from zipfile import ZipFile
import pickle
import pandas as pd

from project2_model import prepare_new_data

base_dir = Path(__file__).resolve().parent

raw_zip_path = base_dir / "project2_raw_data.zip"
model_package_path = base_dir / "data" / "processed" / "packaged_model.pkl"

output_dir = base_dir / "data" / "processed"
classification_output_path = output_dir / "sample_classifications.csv"

# Choose raw monthly file to test, must match name in zip file
selected_txt_file = "2014-07.txt"

# Choose row range
start_row = 5
end_row = 10

def load_rows_from_file(raw_zip_path, selected_txt_file, start_row, end_row):
    sample_rows = []

    if start_row < 1:
        raise ValueError("start_row must be 1 or greater.")

    if end_row < start_row:
        raise ValueError("end_row must be greater than or equal to start_row.")

    with ZipFile(raw_zip_path, "r") as zf:
        txt_files = sorted([name for name in zf.namelist() if name.endswith(".txt")])

        if selected_txt_file not in txt_files:
            raise ValueError(
                f"{selected_txt_file} was not found in the zip file. "
                f"Available files are: {txt_files}")

        with zf.open(selected_txt_file) as f:
            for file_row_number, raw_line in enumerate(f, start=1):
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                if start_row <= file_row_number <= end_row:
                    sample_rows.append((file_row_number, line))

                if file_row_number > end_row:
                    break

    return sample_rows


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_package_path, "rb") as f:
        package = pickle.load(f)

    model = package["model"]
    threshold = package["threshold"]
    feature_columns = package["feature_columns"]
    mapper = package["mapper"]
    model_name = package.get("model_name", "model name not stored")

    sample_rows_with_numbers = load_rows_from_file(
        raw_zip_path=raw_zip_path,
        selected_txt_file=selected_txt_file,
        start_row=start_row,
        end_row=end_row)
    
    if not sample_rows_with_numbers:
        raise ValueError(f"No rows were loaded from {selected_txt_file} "
        f"between rows {start_row} and {end_row}")

    row_numbers = [row_number for row_number, _ in sample_rows_with_numbers]
    sample_rows = [" ".join(row.split()[1:]) for _, row in sample_rows_with_numbers]

    prepared_rows = prepare_new_data(
        raw_rows=sample_rows,
        mapper=mapper,
        feature_columns=feature_columns)

    probabilities = model.predict_proba(prepared_rows)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []

    print("Packaged model:", model_name)
    print("Packaged threshold:", threshold)
    print("Selected source file:", selected_txt_file)
    print(f"Selected row range: {start_row} to {end_row}")
    print("Number of rows tested:", len(sample_rows))
    print("Input rows exclude the original risk score.")
    print()

    for row_number, raw_row, probability, prediction in zip(row_numbers,sample_rows,probabilities,predictions):
        risk_label = "high risk" if prediction == 1 else "low risk"

        results.append({"source_file": selected_txt_file, "row_number": row_number, "raw_input_row": raw_row,
        "predicted_high_risk_probability": round(float(probability), 4), "predicted_class": int(prediction),
        "risk_classification": risk_label})

        print(f"Input row {row_number}:", raw_row)
        print("Predicted high-risk probability:", round(float(probability), 4))
        print("Risk classification:", risk_label)
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv(classification_output_path, index=False)

    print("Saved classifications to:", classification_output_path)

if __name__ == "__main__":
    main()

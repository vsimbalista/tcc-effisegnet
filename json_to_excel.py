import json
import pandas as pd
from openpyxl import load_workbook

def json_to_excel(json_file, excel_file, sheet_name="KFold Results"):
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract metrics
    rows = []
    for fold_data in data:
        fold = fold_data["fold"]
        metrics = fold_data["test_metrics"][0]  # Assume only one set of metrics per fold
        rows.append([
            fold,
            metrics["test_loss"],
            metrics["test_accuracy"],
            metrics["test_recall"],
            metrics["test_precision"],
            metrics["test_iou"],
            metrics["test_f1"]
        ])
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=[
        "N fold", "test_loss", "test_accuracy", "test_recall",
        "test_precision", "test_iou", "test_f1"
    ])
    
    # Load existing Excel file or create a new one
    try:
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Usage
json_to_excel("results/v4_classic_kfold_results_efficientnet-b0_32.json", "results/Effisegnet results.xlsx", sheet_name="v3")

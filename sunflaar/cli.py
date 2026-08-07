import argparse
import logging
import subprocess
import os

def run_dashboard():
    """Launches the Streamlit web dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run(["streamlit", "run", app_path])

def run_prediction(pt_file_path: str):
    """Runs the terminal-based TorchScript inference using NOAA Data."""
    import torch
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sunflaar.data import fetch_goes_data_json
    from sunflaar.plotting import plot_forecast # (Requires your plotting logic if used)

    logging.info(f"Loading compiled model from {pt_file_path} to CPU...")
    device = torch.device("cpu")
    model = torch.jit.load(pt_file_path, map_location=device)
    model.eval()

    df_pivot = fetch_goes_data_json()
    df = df_pivot.resample('5min').mean()
    df.ffill(inplace=True)

    df['Short_Log'] = np.log10(df['Short_0.5_4A'] + 1e-10)
    df['Long_Log'] = np.log10(df['Long_1_8A'] + 1e-10)
    df['Long_Deriv'] = df['Long_Log'].diff().fillna(0.0)

    seq_len = int((12 * 60) / 5)
    latest_df = df.tail(seq_len)
    
    scaled_features = RobustScaler().fit_transform(latest_df[['Short_Log', 'Long_Log', 'Long_Deriv']].values)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(torch.argmax(logits, dim=1).item())
        print(f"\nPREDICTED CLASS: {predicted_class}")

def main():
    parser = argparse.ArgumentParser(description="SunFLAAR Heliophysics Terminal")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command 1: Dashboard
    subparsers.add_parser("dashboard", help="Launch the Heliophysics GUI Terminal")

    # Command 2: Predict
    predict_parser = subparsers.add_parser("predict", help="Run local terminal prediction using TorchScript")
    predict_parser.add_argument("--model", required=True, help="Path to .pt model")

    args = parser.parse_args()

    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "predict":
        run_prediction(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
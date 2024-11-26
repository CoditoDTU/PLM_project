import mlflow
import pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name="Test_experiment"
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    max_results=10000
)

combined_data = []

for run in runs:
    run_name = run.data.tags.get("mlflow.runName")
    
    # Combine metrics and params
    combined_info = {"run_name": run_name}
    combined_info.update(run.data.metrics)
    combined_info.update(run.data.params)
    combined_data.append(combined_info)
    
# Create a DataFrame
combined_df = pd.DataFrame(combined_data)

# Display the DataFrame
print(combined_df)

#bittersweet-rat-286

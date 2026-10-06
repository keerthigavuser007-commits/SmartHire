import pandas as pd
from pathlib import Path

SOURCE = Path("Data/processed/jobs_clean.csv")
OUTPUT = Path("Data/deployment/jobs_deploy.csv")

print("Loading jobs...")

jobs = pd.read_csv(SOURCE)

print("Original jobs:", len(jobs))

# Select 20,000 jobs for deployment
jobs_deploy = jobs.sample(
    n=min(20000, len(jobs)),
    random_state=42
)

jobs_deploy = jobs_deploy.reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

jobs_deploy.to_csv(OUTPUT, index=False)

print("Deployment jobs:", len(jobs_deploy))
print("Saved:", OUTPUT)
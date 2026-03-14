# 50.007 Machine Learning Group Project

## Setup

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh # macOS/Linux

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
    ```

2. Install dependencies

    ```
    uv sync
    uv run pre-commit install
    ```

3. The dataset is too large to be uploaded to GitHub.

    [Download the data set](https://www.kaggle.com/competitions/50-007-machine-learning-spring-2026/data) and put the CSV files in the `data/` directory.

import pandas as pd

def clean_data(file_path):
    data = pd.read_csv(file_path)
    # Drop rows with missing values
    data = data.dropna()
    # remove header, and column "country" and save file
    data.to_csv(file_path, header=False, index=False, columns=["child_mort", "exports",
        "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"])
    
if __name__ == "__main__":
    clean_data("data.csv")
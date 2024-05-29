import pandas as pd

columns = ["child_mort","exports","health","imports","income","inflation","life_expec","total_fer","gdpp"]


def clean_data(file_path):
    data = pd.read_csv(file_path)
    # Drop rows with missing values
    data = data.dropna()
    # remove header, and column "country" and save file
    data.to_csv(file_path, header=False, index=False, columns=columns)
    
def splitData(file_path):
    data = pd.read_csv(file_path)
    data.dropna()
    data.drop(columns=["country"], inplace=True)
    print(data.head())
    print(data[columns[:2]].head())
    for i in range(2, len(columns)):
        file_name = "data_" + str(i - 1) + ".csv"
        data[columns[:i]].to_csv(file_name, header=False, index=False)
    
def read_array_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Remove leading and trailing whitespace and brackets
        content = content.strip('[]')
        # Split the content into individual elements
        array = [int(x) for x in content.split(',') if x.strip()]
        arr = []
        for index in range(0, len(array)):
            if index % 2 == 0:
                arr.append(array[index]/1000)
    return arr
    
    
    
if __name__ == "__main__":
    seqTimes = read_array_from_file("seqTimes.txt")
    cudaTimes = read_array_from_file("cudaTimes.txt")
    ompTimes = read_array_from_file("ompTimes.txt")
    print("Sequential Times:\n", seqTimes)
    print("OMP Times:\n", ompTimes)
    print("CUDA Times:\n", cudaTimes)
    ompDiff = [round(seqTimes[i] - ompTimes[i], 3)  for i in range(0, len(seqTimes))]
    cudaDiff = [round(seqTimes[i] - cudaTimes[i], 3) for i in range(0, len(seqTimes))]
    print("OMP Diff:\n", ompDiff)
    print("CUDA Diff:\n", cudaDiff)
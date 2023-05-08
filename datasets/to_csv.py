# dataset name, access from local folder
name = "year/YearPredictionMSD_whole"

fin = open(name)
fout = open(name + ".csv", "w") # make a csv from the dataset

### change number of feature d depending on the dataset
if name.startswith("year"):
    d = 90
elif name.startswith("cov"):
    d = 54
elif name.startswith("ijcnn1"):
    d = 22

fout.write(f"label,{','.join(['d'+str(i) for i in range(1,d+1)])}\n") # put heading in the first line
for line in fin.readlines():
    line = line[:-1]
    label = line.split()[0]
    datapoint_dict = {token.split(":")[0]:token.split(":")[1] for token in line.split()[1:]}
    datapoint = ["" if str(i) not in datapoint_dict else datapoint_dict[str(i)] for i in range(1,d+1)]

    # write label in first column, then features
    fout.write(f"{label},{','.join(datapoint)}\n")
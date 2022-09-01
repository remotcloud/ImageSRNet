import os
dir=r"V:\yang\code\Sesmic\ImageSRNet\Result\CGP\SplitLessImage\Fitness"
dirs = os.listdir(dir)
resultPath = r"V:\yang\code\Sesmic\ImageSRNet\Result\CGP\SplitLessImage"
resultFile=os.path.join(resultPath,"bestFitness.txt")
for file in dirs:
    filePath=os.path.join(dir, file)
    dataNum=file.split(".")[0]
    lastLine=""
    bestFitness=""
    print(file)
    print(dataNum)
    with open(resultFile, "a") as rf:
        with open(filePath, "r") as f:
            lines = f.readlines()
            lastLine = lines[-1]
            bestFitness = lastLine.split(" ")[-1].strip("tensor(")
            bestFitness = bestFitness.replace(")", "")
        rf.write(dataNum+" "+bestFitness)



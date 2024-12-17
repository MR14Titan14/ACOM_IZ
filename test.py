import time
from IZ import predict_roboflow

start=time.time()
counter=0
with open("test/labels.txt","r") as file:
    for line in file:
        name=line.split("-")[0]
        label=line.split("-")[1]
        res,_=predict_roboflow(f"test/{name}.jpg",0.40,0.75)
        label=label.replace("\n","")
        print(name)
        print(f"label: {label}")
        print(f"res: {res}")
        if(res==label):
            counter+=1
print(counter)
end=time.time()
print(end-start)
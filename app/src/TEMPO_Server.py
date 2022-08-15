import socket, threading
from PIL import Image
import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import numpy as np
import tensorflow as tf
import pymongo
from datetime import datetime



class Images():
    def __init__(self):
        self.items = []
        self.crystal = []
        self.final_prices = []
        self.crystalP = 0
        
    def getJson(self):
        self.Json = {
    "id" : "TEMPO",
    "crystal" : self.crystalP,
    "i1" : {
        "name" : "Destruction Stone Fragment",
        "price" : self.final_prices[0]
    },
    "i2" : {
        "name" : "Guardian Stone Fragment",
        "price" : self.final_prices[1]
    },
    "i3" : {
        "name" : "Destruction Stone",
        "price" : self.final_prices[2]
    },
    "i4" : {
        "name" : "Guardian Stone",
        "price" : self.final_prices[3]
    },
    "i5" : {
        "name" : "Guardian Stone Crystal",
        "price" : self.final_prices[4]
    },
    "i6" : {
        "name" : "Destruction Stone Crystal",
        "price" : self.final_prices[5]
    },
    "i7" : {
        "name" : "Simple Oreha Fusion Material",
        "price" : self.final_prices[6]
    },
    "i8" : {
        "name" : "Harmony Shard Pouch (S)",
        "price" : self.final_prices[7]
    },
    "i9" : {
        "name" : "Life Shard Pouch (S)",
        "price" : self.final_prices[8]
    },
    "i10" : {
        "name" : "Solar Grace",
        "price" : self.final_prices[9]
    },
    "i11" : {
        "name" : "Honor Shard Pouch (S)",
        "price" : self.final_prices[10]
    },
    "i12" : {
        "name" : "Caldarr Fusion Material",
        "price" : self.final_prices[11]
    },
    "i13" : {
        "name" : "Basic Oreha Fusion Material",
        "price" : self.final_prices[12]
    },
    "i14" : {
        "name" : "Harmony Leapstone",
        "price" : self.final_prices[13]
    },
    "i15" : {
        "name" : "Harmony Shard Pouch (M)",
        "price" : self.final_prices[14]
    },
    "i16" : {
        "name" : "Life Leapstone",
        "price" : self.final_prices[15]
    },
    "i17" : {
        "name" : "Life Shard Pouch (M)",
        "price" : self.final_prices[16]
    },
    "i18" : {
        "name" : "Honor Leapstone",
        "price" : self.final_prices[17]
    },
    "i19" : {
        "name" : "Honor Shard Pouch (M)",
        "price" : self.final_prices[18]
    },
    "i20" : {
        "name" : "Solar Blessing",
        "price" : self.final_prices[19]
    },
    "i21" : {
        "name" : "Great Honor Leapstone",
        "price" : self.final_prices[20]
    },
    "i22" : {
        "name" : "Star's Breath",
        "price" : self.final_prices[21]
    },
    "i23" : {
        "name" : "Harmony Shard Pouch (L)",
        "price" : self.final_prices[22]
    },
    "i24" : {
        "name" : "Moon's Breath",
        "price" : self.final_prices[23]
    },
    "i25" : {
        "name" : "Life Shard Pouch (L)",
        "price" : self.final_prices[24]
    },
    "i26" : {
        "name" : "Honor Shard Pouch (L)",
        "price" : self.final_prices[25]
    },
    "i27" : {
        "name" : "Solar Protection",
        "price" : self.final_prices[26]
    }
}



class LAOPP():
    def __init__(self):
        self.device = torch.device('cpu')
        self.batch_size = 100
        self.num_classes = 10 # 0~9
        self.cx_st = 53
        self.cx_ed = 65
        self.ix_st = 45
        self.ix_ed = 56
        self.model = torch.load("./LAO.pth",map_location=self.device)
        
    def resizing(self,img):
        img = Image.fromarray(img)
        img = img.resize(size=(28, 28))
        img = np.array(img)
        img[img<=50] = 0 #threadholding = 50
        return img
    
    def predict(self,data):
        data = data.to(self.device)
        out = self.model(data)
        pred = torch.max(out.data, 1)[1]
        return pred
        
    def getPrice(self,img,TYPE):
        if(TYPE == "C"):
            x_st = self.cx_st
            x_ed = self.cx_ed
        elif(TYPE == "I"):
            x_st = self.ix_st
            x_ed = self.ix_ed
        else:
            print("type error")
            
        price_array = torch.zeros(4,1,28,28)
        digit = 0
        gap = x_ed - x_st
        for i in range(4):
            im = self.resizing(img[:,x_st - i*gap : x_ed - i*gap])
            if np.sum(im) == 0.0:
                break
            else:
                im = im/255
                digit += 1
                price_array[i,0,:,:] = torch.from_numpy(im)

        value = self.predict(price_array[:digit,:,:,:]).tolist()
        price = ""
        
        for d in value[-1::-1]:
            price += str(d)
            
        return int(price)



#CNN model structure
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 10, 5, padding = 2),
      nn.BatchNorm2d(10),
      nn.ReLU(inplace=True),
      nn.Conv2d(10, 20, 5),
      nn.BatchNorm2d(20),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(20, 40, 5, padding = 2),
      nn.BatchNorm2d(40),
      nn.ReLU(inplace=True),
      nn.Conv2d(40, 80, 5),
      nn.BatchNorm2d(80),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),
    )

    self.fc_net = nn.Sequential(
      nn.Dropout2d(p=0.25, inplace=False),
      nn.Linear(80 * 4 * 4, 100),
      nn.BatchNorm1d(100),
      nn.ReLU(inplace=True),
      nn.Linear(100, num_classes),
    ) # channel * W * H for first linear(FC) parameter

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(-1, 80 * 4 * 4) #2D -> flat
    x = self.fc_net(x)

    return F.log_softmax(x)


class TCPServerThread(threading.Thread):
    def __init__(self,tcpServerThreads,connections,connection,clientAddress,image_obj, timestamp):
        threading.Thread.__init__(self)
        
        self.tcpServerThreads = tcpServerThreads
        self.connection = connection
        self.connections = connections
        self.clientAddress = clientAddress
        self.num_items = 27
        self.images = [0 for i in range(self.num_items)]
        self.crystal = None
        self.image_obj = image_obj
        self.timestamp = timestamp
        
    def run(self):
        try:
            greeting = self.connection.recv(1024)
            if int(datetime.now().timestamp()) > self.timestamp[0] + 30*60:
                if(greeting.decode() == "hello"):
                    self.connection.send("ok".encode())
                    
                for i in range(self.num_items):
                    self.images[i] = self.connection.recv(65536)
                    self.connection.send("ok".encode())
                self.crystal = self.connection.recv(65536)
                self.connection.send("ok".encode())

                for i in range(self.num_items):
                    nparr = np.frombuffer(self.images[i],np.uint8)
                    self.images[i] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[21:35,:,0]
                nparr = np.frombuffer(self.crystal,np.uint8)
                self.crystal = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:,:,0]

                self.image_obj.items = self.images
                self.image_obj.crystal = self.crystal

                pp = LAOPP()
                pp.model.eval()
                for i in range(len(self.image_obj.items)):
                    self.image_obj.final_prices.append(pp.getPrice(self.image_obj.items[i],"I"))
                self.image_obj.crystalP = pp.getPrice(self.image_obj.crystal,"C")
                self.image_obj.getJson()



                #print(self.image_obj.final_prices)
                #print(self.image_obj.crystalP)

                f = open("./pydb.txt", 'r')
                dbcode = f.readline()
                f.close()

                client = pymongo.MongoClient(dbcode)
                #print(client.stats)
                #for db in client.list_databases():
                #   print(db)
                db = client.LAO
                #for collection in db.list_collection_names():
                #    print(collection)
                collection = db.maris

                collection.delete_many( { 'id': "TEMPO" } )
                collection.insert_one(self.image_obj.Json).inserted_id
                self.timestamp[0] = int(datetime.now().timestamp())
            else:
                print()
                print("data is updated within 30 mins")
                if(greeting.decode() == "hello"):
                    self.connection.send("exit".encode())
        
        except Exception as error:
            print(error)
            self.connections.remove(self.connection)
            self.tcpServerThreads.remove(self)
            exit(0)
        self.connections.remove(self.connection)
        self.tcpServerThreads.remove(self)


class TCPServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.host = ""
        self.port = 4000
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.bind((self.host,self.port))
        self.serverSocket.listen(1)
        
        self.connections= []
        self.tcpServerThreads = []
        
        self.timestamp = [0] #int(datetime.now().timestamp())
        
    def run(self):
        
        try:
            while True:
                print("server is waiting")
                connection, clientAddress = self.serverSocket.accept()
                self.connections.append(connection)
                print("client connected")
                
                image_obj = Images()
                subThread = TCPServerThread(self.tcpServerThreads,self.connections, connection, clientAddress,image_obj, self.timestamp)
                self.tcpServerThreads.append(subThread)
                subThread.start()
                
                
        except Exception as error:
            print(error)
    
    def sendAll(self,message):
        try:
            self.tcpServerThreads[0].send(message)
        except:
            pass


if __name__ == '__main__':
    Server = TCPServer()
    Server.start()





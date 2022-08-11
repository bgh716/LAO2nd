import socket
import sys
from PIL import Image
import os
import io
from array import array
import base64
import cv2
import numpy as np
from tkinter import Tk, Label, Button
from tkinter import filedialog
import webbrowser


class client:
    def __init__(self):
        self.host = "34.125.169.162"
        self.port = 4000
        self.addr = (self.host,self.port)
        self.pr = ['1.height: 1440','2.width: 2560']
    def run(self,indexed_pic,crystal):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect(self.addr)
            except Exception as error:
                print(error)
                sys.exit()
                return 2

            sock.send("hello".encode())
                
            for i in range(len(indexed_pic)):
                image = cv2.imencode(".jpg",indexed_pic[i])[1].tostring()
                resp = sock.recv(1024)
                if resp.decode() == "ok":
                    sock.send(image)
                elif resp.decode() == "exit":
                    sock.close()
                    print("updated within 30 mins")
                    return 1
                    
            image = cv2.imencode(".jpg",crystal)[1].tostring()
            resp = sock.recv(1024)
            if resp.decode() == "ok":
                sock.send(image)
            resp = sock.recv(1024)
            if resp.decode() == "ok":
                print("Success")
                sock.close()
                return 0


class image:
    def __init__(self,height,width,PATH,PATH_image):
        self.auction_pages = 4
        self.crystal_page = 1
        self.num_screenshots = self.auction_pages + self.crystal_page #4 auction pages + 1 crystal price page
        self.num_items = 10
        self.items_to_be_evaluated = 27
        self.height = height
        self.width = width
        
        self.PATH = PATH
        self.PATH_image = PATH_image
        self.itemTable = np.load(os.path.join(PATH,"itemTable.npy"))
        
        self.pics = [f for f in os.listdir(PATH_image) if os.path.isfile(os.path.join(PATH_image, f))]
        self.pics.sort(reverse = True) #most recent file first
        self.pics = self.pics[:self.num_screenshots]
        self.pics.sort()
        
        self.val_in = [0 for i in range(self.auction_pages*self.num_items)]
        self.prices = [0 for i in range(self.num_items)]
        self.crystal = None
        self.indexed_pic = [0 for i in range(self.items_to_be_evaluated)]
        
    def imageProcessing(self):
        for index,file in enumerate(self.pics):
            img = cv2.imread(os.path.join(self.PATH_image,file))
            if index<self.auction_pages:
                self.val_in[index*10:((index+1)*10)] = self.get_item_id(img , self.itemTable)
                self.prices[index*10:((index+1)*10)] = self.get_prices(img)
            else:
                self.crystal = self.get_crystal_price(img)
            
        min_val = []
        min_val = [0 for i in range(self.items_to_be_evaluated)]
        for i in range(self.auction_pages*self.num_items):
            index = int(self.val_in[i][1])
            value = int(self.val_in[i][0])
            if min_val[index] == 0 or min_val[index] > value:
                min_val[index] = value
                self.indexed_pic[index] = self.prices[i]
        
    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
    def get_item_val(self,itemTable,itemColor):
        idx = 0
        for i in range(27):
            itemCode = itemTable[i]
            val = ((itemCode[0] - itemColor[0])**2 + (itemCode[1] - itemColor[1])**2 + (itemCode[2] - itemColor[2])**2)**(0.5)
            if i == 0:
                min = val
            elif val < min:
                idx = i
                min = val
        return [min,idx]
    
    def get_item_id(self,img,itemTable):
        y_st = 484 #starting y coordinate
        y_ed = 541 #ending y coordinate
        y_gap = y_ed - y_st
        val_in=[]
        val_in = [0 for i in range(self.num_items)] 
        for i in range(self.num_items):
            val_in[i] = self.get_item_val(itemTable,[np.sum(img[y_st:y_ed,873:925,0]),np.sum(img[y_st:y_ed,873:925,1]),np.sum(img[y_st:y_ed,873:925,2])])
            y_st += y_gap
            y_ed += y_gap
        return val_in
    
    def get_prices(self,img):
        prices = []
        prices = [0 for i in range(self.num_items)] 
        y_st = 484 #starting y coordinate
        y_ed = 541 #ending y coordinate
        y_gap = y_ed - y_st
        x_st = 1463 #recent price starting x coordinate
        x_ed = 1519 #recent price ending x coordinate
        #x_st = 1668 #lowest price
        #x_ed = 1680 #lowest price
        for i in range(self.num_items): # 10 items per page
            im = img[y_st:y_ed,x_st:x_ed] #crop image
            im = self.rgb2gray(im)
            im[im<=50] = 0
            prices[i] = im
            y_st = y_ed
            y_ed += y_gap
        return prices
    
    def get_crystal_price(self,img):
        y_st = 833
        y_ed = 851
        x_st = 1643
        x_ed = 1710
        x_gap = x_ed - x_st - 1
        im = img[y_st:y_ed,x_st:x_ed]
        im = self.rgb2gray(im)
        im[im<=50] = 0
        return im


def set_dir():
    root.dirName = filedialog.askdirectory()
    text.configure(text=root.dirName)
    f = open("dir.txt",'w')
    data = root.dirName
    f.write(data)
    f.close()
    
        
        

def send():
    items = image(1440,2560,os.getcwd(),os.path.normpath(root.dirName))
    items.imageProcessing()
    if(request.run(items.indexed_pic,items.crystal)==0):
        res_label.configure(text="Success")
    elif(request.run(items.indexed_pic,items.crystal)==1):
        res_label.configure(text="updated within 30 mins")
    elif(request.run(items.indexed_pic,items.crystal)==2):
        res_label.configure(text="Server is offline")

def callback(url):
    webbrowser.open_new(url)

if __name__ == '__main__':
    request = client()
    if os.path.isfile(os.path.join(os.getcwd(),"dir.txt")):
        f = open("dir.txt",'r')
        direct = f.readline()
        f.close()
    else:
        direct = ""
    root = Tk()
    root.title("TEMPO")
    root.geometry("540x150")
    label = Label(root,text="Set Directory")
    label.pack()
    text = Label(root,text="")
    text.pack()
    if direct != "":
        root.dirName = direct
        text.configure(text=direct)
    set_button = Button(root, text="Set", command = set_dir)
    set_button.pack()
    send_button = Button(root,text="Send", command = send)
    send_button.pack()
    res_label = Label(root,text="")
    res_label.pack()
    link_label = Label(root,text="https://marishop.herokuapp.com/shop", fg="blue", cursor="hand2")
    link_label.pack()
    link_label.bind("<Button-1>", lambda e: callback("https://marishop.herokuapp.com/shop"))
    root.mainloop()






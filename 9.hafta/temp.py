import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/Users/mehmetcanak/Desktop/g/9.hafta/" 
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",") 
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

print(train_data[600,0])

m,n=train_data.shape
print(m,n)

im5=train_data[4,1:]
im7=im5.reshape(28,28)

plt.imshow(im7,cmap='gray')
plt.show()

#liste={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
#liste.keys()

def cluster_centroid_for_mnist(train_data):
    m,n=train_data.shape
    toplam,sutunSayisi=0,0
    geneltoplam=0
    sgenel=0
    ortalama=0
    liste={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    satirSayisi={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    listeSonuc={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for i in range(m):
        for j in range(1,n):
            toplam=toplam+train_data[i,j]
            sutunSayisi=sutunSayisi+1
       # if(train_data[i,0]==liste.keys()):
        ortalama=toplam/sutunSayisi
        liste[train_data[i,0]]=liste[train_data[i,0]]+ortalama
        satirSayisi[train_data[i,0]]=satirSayisi[train_data[i,0]]+1
        sutunSayisi=0
        toplam=0
    for i in range(10):
        listeSonuc[i]=liste[i]/satirSayisi[i]
    return listeSonuc



liste=cluster_centroid_for_mnist(train_data)
liste

def cluster_centroid_for_img(im):
    m,n=im.shape
    toplam,ort=0,0
    for i in range(m):
        for j in range(n):
            if(float(im[i,j]!=0)):
                im[i,j]=im[i,j]*100
               # print(im[i,j])
            toplam=toplam+im[i,j]
    ort=toplam/n
    return ort


my_test_img=plt.imread('test13.PNG')
plt.imshow(my_test_img,cmap='gray')
plt.show()

my_test_img.shape
im2=my_test_img[0:28,0:28,0]
im2.shape

im3=im2.reshape(1,784)
im3.shape

plt.imshow(im2,cmap='gray')
plt.show()

plt.imshow(im3,cmap='gray')
plt.show()

liste2=cluster_centroid_for_img(im3)
liste2


mutlakDeger1=abs(liste2-liste[0])
kume=0
for i in range(1,10):
    mutlakDeger2=abs(liste2-liste[i])
    if(mutlakDeger1<mutlakDeger2):
        mutlakDeger=mutlakDeger1
        mutlakDeger1=mutlakDeger2
        kume=i
        
    
print(kume," kumesine aittir.(olasılıksal olarak en yuksektir)")



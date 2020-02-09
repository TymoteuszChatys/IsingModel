import numpy as np;
import matplotlib.pyplot as plt
import random
import os
import time
import pandas as pd
from PIL import Image
from datetime import datetime
from datetime import timedelta



# plot average M against T
# average E against T

choice = 0



def CheckOutputFiles():
    for n in np.arange(1,1000000,1):
        if os.path.exists('outputs/output%s' %(n),)==False:
            os.makedirs(('outputs/output%s' %(str(n),)), exist_ok=True)
            return n

def CheckForSplit(lattice,x,y,z): 
    a = True
    b = True
    c = True
    for k in np.arange(0,z,1):
        nlattice = lattice[0:z,0:x,k:k+1]
        if np.abs(np.mean(nlattice)) !=1:
            c = False 
            k = z
    for j in np.arange(0,y,1):
        nlattice = lattice[j:j+1,0:x,0:y]
        if np.abs(np.mean(nlattice)) !=1:
            a = False 
            j = y
    for i in np.arange(0,x,1):
        nlattice = lattice[0:z,i:i+1,0:y]
        if np.abs(np.mean(nlattice)) !=1: 
            b = False
            i = x
    if a == False and b == False and c == False:
        return False
    else:
        return True

def SaveImages(x,y,i,p,lattice,z,n):
    if z==0:
        a=display(lattice)
        a.save('outputs/output%s/Image0.bmp' %(str(n)))
    if z==1:
        a=display(lattice)
        a.save("outputs/output{0}/Image{1}.bmp" .format(str(n), int(i/p)))
        
def SaveImages2(x,y,p,lattice,z,n):
    if z==0:
        a=display(lattice)
        a.save('outputs/output%s/Image0.bmp' %(str(n)))
    if z==1:
        a=display(lattice)
        a.save("outputs/output{0}/Image{1}.bmp" .format(str(n), int(p)))
    
def display(field):
    return Image.fromarray(np.uint8((field)*0.5*500))
    
def InitialLattice(x,y,z):
    #Generates a lattice of spins. Dimensions x*y
    ilattice = (np.random.randint(2, size=(x,y,z))*2)-1
    return ilattice
  
def TotalMagnitude(x):
    #Summates all the spins to get the total Magnetization of the lattice
    magnitude = np.sum(x)
    return magnitude 
      
def TotalEnergy(lattice,x,y,z,Jd):
    energy = 0
    for i in np.arange(0,x,1):
        for j in np.arange(0,y,1):
            for k in np.arange(0,z,1):    
                currentpoint = lattice[i,j,k]
                neighbours = lattice[(i+1)%x, j,k] + lattice[(i-1)%x, j,k] + lattice[i, (j+1)%y,k] + lattice[i, (j-1)%y,k] + lattice[i, j,(k-1)%z] + lattice[i, j,(k+1)%z] + Jd*(lattice[(i-1)%x, (j-1)%y,k] + lattice[(i+1)%x, (j-1)%y,k] + lattice[(i+1)%x, (j+1)%y,k] + lattice[(i-1)%x, (j+1)%y,k] + lattice[(i-1)%x, (j+1)%y,k] + lattice[(i, (j+1)%y,(k+1)%z)] + lattice[(i, (j+1)%y,(k-1)%z)]  + lattice[(i, (j-1)%y,(k+1)%z)] + lattice[(i, (j-1)%y,(k-1)%z)] + lattice[(i-1)%x, j,(k-1)%z] + lattice[(i-1)%x, j,(k+1)%z] + lattice[(i+1)%x, j,(k-1)%z]+ lattice[(i+1)%x, j,(k+1)%z])
                energy += -currentpoint*neighbours
    return energy/2
    #Gives the energy of a given lattice
 
def Energy(lattice,i,j,k,x,y,z,Jd):
    currentpoint = lattice[i,j,k]
    neighbours = lattice[(i+1)%x, j,k] + lattice[(i-1)%x, j,k] + lattice[i, (j+1)%y,k] + lattice[i, (j-1)%y,k] + lattice[i, j,(k-1)%z] + lattice[i, j,(k+1)%z] + Jd*(lattice[(i-1)%x, (j-1)%y,k] + lattice[(i+1)%x, (j-1)%y,k] + lattice[(i+1)%x, (j+1)%y,k] + lattice[(i-1)%x, (j+1)%y,k] + lattice[(i-1)%x, (j+1)%y,k] + lattice[(i, (j+1)%y,(k+1)%z)] + lattice[(i, (j+1)%y,(k-1)%z)]  + lattice[(i, (j-1)%y,(k+1)%z)] + lattice[(i, (j-1)%y,(k-1)%z)] + lattice[(i-1)%x, j,(k-1)%z] + lattice[(i-1)%x, j,(k+1)%z] + lattice[(i+1)%x, j,(k-1)%z]+ lattice[(i+1)%x, j,(k+1)%z])
    energy = -currentpoint*neighbours
    return energy

def Metropolis(lattice,x,y,z,T,Jd):
    #Applies the Metropolish algorithm
    #Pick a random point in the lattice
    chosenx = np.random.randint(0,x)
    choseny = np.random.randint(0,y)
    chosenz = np.random.randint(0,z)
    iE = Energy(lattice,chosenx,choseny,chosenz,x,y,z,Jd)
    spin = lattice[chosenx,choseny,chosenz]
    dE = (-iE)-iE
    if dE>0:
        if np.exp(-((dE)/T)) > random.uniform(0, 1):
            lattice[chosenx,choseny,chosenz]=-spin
    else:
        lattice[chosenx,choseny,chosenz]=-spin
    return lattice

def Alg1():
    T=1
    print("Choosing Lattice Dimensions")
    x = int(input("x:"))
    y = int(input("y:"))
    z = int(input("z:"))
    images = int(input("Do you want to save images in the output folder (Y=1,N=0)"))
    fast = int(input("Fast mode? (doesn't check if finished) (Y=1,N=0)"))
    if fast == 1:
        Alg3(images,x,y,z)
    elif fast == 0:
        Alg2(images,x,y,z,T,1)

def Alg2(images,x,y,z,T,finish,Jd):
    lattice = InitialLattice(x,y,z)
#    mag = np.array([])
    steps = 10000
    i=0
    p=5
    full = 0
    TotalM=np.array([])
    TotalE=np.array([])
    if images == 1:
        n = CheckOutputFiles()
        SaveImages(x,y,z,0,p,lattice,0,n)
    if finish == 1:
        while CheckForSplit(lattice,x,y,z) == False:
            i=i+1
            lattice = Metropolis(lattice,x,y,z,T,Jd)
            totalMag = TotalMagnitude(lattice)
            TotalM = np.append(TotalM,totalMag)
            TotalEne = TotalEnergy(lattice,x,y,z,Jd)
            TotalE = np.append(TotalE,TotalEne)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,z,i,p,lattice,1,n)
    elif finish == 0:
        for j in np.arange(0,steps,1):
            i=i+1
            lattice = Metropolis(lattice,x,y,z,T,Jd)
            totalMag = TotalMagnitude(lattice)
            TotalM = np.append(TotalM,totalMag)
            TotalEne = TotalEnergy(lattice,x,y,z,Jd)
            TotalE = np.append(TotalE,TotalEne)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,z,i,p,lattice,1,n)
    if images == 1:
        SaveImages(x,y,z,i,p,lattice,1,n)
    if np.abs(TotalMagnitude(lattice)) == x*y*z:
        full = 1
    avgM = np.mean(TotalM)
    avgE = np.mean(TotalE)
    return i, avgE, avgM, full

def Alg3(images,x,y,z):
    lattice = InitialLattice(x,y,z)
#    mag = np.array([])
    T=3
    l=0
    i=0
    if images == 1:
        p=0
        n = CheckOutputFiles()
        SaveImages2(x,y,z,p,lattice,0,n)
    while i == 0:
        for a in np.arange(0,x*y*z,1):
            lattice = Metropolis(lattice,x,y,z,T)
        l=l+1
        SaveImages2(x,y,z,l,lattice,1,n)        
    if images == 1:
        SaveImages(x,y,z,i,p,lattice,1,n)
    return i

def Alg4lim(x):
    if 0<x<=4:
        limit = 1000
    if 4<x<=8:
        limit = 10000
    if 8<x<=12:
        limit = 100000
    if 12<x<=14:
        limit = 150000
    if x == 15:
        limit = 200000
    return limit


def Alg4(images,x,y,z,T,finish,Jd):
    lattice = InitialLattice(x,y,z)
#    mag = np.array([])
    steps = 1000
    i=0
    p=5
    full = 0
    if images == 1:
        n = CheckOutputFiles()
        SaveImages(x,y,z,0,p,lattice,0,n)
    if finish == 1:
        while CheckForSplit(lattice,x,y,z) == False and i<Alg4lim(x):
            i=i+1
            lattice = Metropolis(lattice,x,y,z,T,Jd)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,z,i,p,lattice,1,n)
    elif finish == 0:
        for j in np.arange(0,steps,1):
            i=i+1
            lattice = Metropolis(lattice,x,y,z,T,Jd)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,z,i,p,lattice,1,n)
    if images == 1:
        SaveImages(x,y,z,i,p,lattice,1,n)
    if np.abs(TotalMagnitude(lattice)) == x*y*z:
        full = 1
    #print(i, "for lat " , x)
    return i, full

def StepsToFinish():
    print(datetime.now())
    T=0.5
    minlattice = 3
    maxlattice = 3
    iterations = 500
    Jd = 0.3
    fulls=np.array([])
    for i in np.arange(0,iterations,1):
        startw = time.time()
        latticesize= np.array([])
        numbersteps= np.array([])
        timea=np.array([])
        ttimea=np.array([])
        averageE=np.array([])
        averageM=np.array([])
        temperature=np.array([])
        x=np.array([])
        for j in np.arange(minlattice,maxlattice+1,1): 
            start = time.time()
            temperature = np.append(temperature, T)
            latticesize = np.append(latticesize, j)
            b, avgE, avgM, full= Alg2(0,j,j,j,T,1,Jd)
            fulls = np.append(fulls, full)
            numbersteps = np.append(numbersteps, b)
            averageE = np.append(averageE, avgE)
            averageM = np.append(averageM, avgM)
            x = np.append(x,j)
            end = time.time()
            totaltime = end-start
            timea = np.append(timea,totaltime)
            #print(j ,"took " , totaltime , " seconds and ", b, " steps")
        endw = time.time()
        totaltime = (endw-startw)
        ttimea = np.append(ttimea,totaltime)
        totalestimatedtime = (np.mean(ttimea))*(iterations-i)
        finaltime = datetime.now()+timedelta(seconds=totalestimatedtime)
        #print("Total iteration time ", totaltime, " seconds. Finish ETA :", finaltime.isoformat(timespec='seconds'))
        SavetoCSV(latticesize,numbersteps,timea,averageE/(x*x*x),averageM/(x*x*x),temperature)
        del latticesize, numbersteps, timea, averageE, averageM, temperature
    print("Percentage full: ", np.sum(fulls)*100/fulls.size, "using ", fulls.size, "simulations")
    print(datetime.now())
    
def ChangeJ():
    print(datetime.now())
    T=0.1
    minJd = 0
    maxJd = 0.7
    step = 0.7
    iterations = 100
    for k in np.arange(9,10,1):
        Jd=np.array([])
        percentages=np.array([])
        latsizes= np.array([])
        for j in np.arange(minJd,maxJd+step,step):
            startw = time.time()
            numbersteps= np.array([])
            timea=np.array([])
            ttimea=np.array([])
            x=np.array([])
            fulls=np.array([])
            tttimea = np.array([])
            for i in np.arange(0,iterations,1): 
                start = time.time()
                b, full= Alg4(0,k,k,k,T,1,j)
                fulls = np.append(fulls, full)
                numbersteps = np.append(numbersteps, b)
                x = np.append(x,j)
                end = time.time()
                totaltime = end-start
                tttimea = np.append(tttimea,totaltime)
                if i%100==0:      
                   totalestimatedtime = (np.mean(tttimea))*((iterations-i)/1)
                   finaltime = datetime.now()+timedelta(seconds=totalestimatedtime)
                   print("Finish ETA :", finaltime.isoformat(timespec='seconds'))
                if full==1:
                    print(j ,"took " , totaltime , " seconds and ", b, " steps")
            percent = np.sum(fulls)*100/fulls.size
            endw = time.time()
            totaltime = (endw-startw)
            ttimea = np.append(ttimea,totaltime)
            totalestimatedtime = (np.mean(ttimea))*((maxJd-j)/step)
            finaltime = datetime.now()+timedelta(seconds=totalestimatedtime)
            print("Total iteration time ", totaltime, " seconds. Finish ETA :", finaltime.isoformat(timespec='seconds'))
            Jd= np.append(Jd, j)
            latsizes = np.append(latsizes,k)
            percentages = np.append(percentages,percent)
            print("Percentage full: ", percent, "using ", fulls.size, "simulations for ", j, "lattice: ",k)
            del numbersteps, timea,fulls
        SavetoCSV2(Jd,percentages,k)
        print(datetime.now())

def Temperature():
    print(datetime.now())
    x = y = z = 5
    maxT = 7
    minT = 0.5
    step = 0.5
    iterations = 1
    Jd = 0.5
    for i in np.arange(0,iterations,1):
        startw = time.time()
        latticesize= np.array([])
        numbersteps= np.array([])
        timea=np.array([])
        ttimea=np.array([])
        temperature=np.array([])
        averageE=np.array([])
        averageM=np.array([])
        for T in np.arange(minT,maxT+step,step): 
            start = time.time()
            latticesize = np.append(latticesize, x)
            temperature = np.append(temperature, T)
            steps, avgE, avgM, full= Alg2(0,x,y,z,T,0,Jd)
            averageE = np.append(averageE, avgE)
            averageM = np.append(averageM, avgM)
            numbersteps = np.append(numbersteps, steps)
            end = time.time()
            totaltime = end-start
            timea = np.append(timea,totaltime)
            print(T ,"took " , totaltime , " seconds and ", steps, " steps")
        b = plt.scatter(temperature,np.abs(averageE/(x*y*z)))
        c = plt.scatter(temperature,np.abs(averageM/(x*y*z)))
        SavetoCSV(latticesize,numbersteps,timea,averageE/(x*y*z),averageM/(x*y*z),temperature)
        del latticesize, numbersteps, timea, averageE, averageM,temperature
        endw = time.time()
        totaltime = (endw-startw)
        ttimea = np.append(ttimea,totaltime)
        totalestimatedtime = (np.mean(ttimea))*(iterations-i-1)
        finaltime = datetime.now()+timedelta(seconds=totalestimatedtime)
        print("Total iteration time ", totaltime, " seconds. Finish ETA :", finaltime.isoformat(timespec='seconds'))
    b.show()
    c.show()
    print(datetime.now())

def SavetoCSV(lattice,steps,timea,averageE,averageM,T):
    df = pd.DataFrame({"latticesize" : lattice, "numbersteps" : steps, "totaltime" : timea, "averageE" :np.abs(averageE), "averageM" :np.abs(averageM), "temperature" :T})
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("results.csv", index=False, mode = 'a')
    
def SavetoCSV2(Jd,full,latticesize):
    df = pd.DataFrame({"Jd" : Jd, "percent" : full, "lattice" : latticesize})
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("resultsJd.csv", index=False, mode = 'a')
    
    
print("Ising Model - THEORY PROJECT")
while choice != 8:
    print("1 - |Steps to Finish- Square Lattices|")
    print("2 - |Temperature Change|")
    print("3 - |1 Simulation|")
    print("4 - |J-Dash Change|")
    print("6 - |Initial Magnetization Distribtuion|")
    print("7 - |Initial Energy Distribtuion|")
    print("8 - |Exit|")
    choice = int(input("Choose Option : "))
    if choice == 1:
        start = time.time()
        StepsToFinish()
        end = time.time()
        print("Total simulation time ", end-start, " seconds.")
    elif choice == 3:
        Alg1()
    elif choice == 2:
        start = time.time()
        Temperature()
        end = time.time()
        print("Total simulation time ", end-start, " seconds.")
    elif choice == 4:
        start = time.time()
        ChangeJ()
        end = time.time()
        print("Total simulation time ", end-start, " seconds.")
        
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
        
def CheckOutputFiles2():
    for n in np.arange(1,1000000,1):
        if os.path.isfile('outputs2/Image%s.bmp' %(n),)==False:
            return n

def CheckForSplit(lattice,x,y): 
    a = True
    b = True
    for j in np.arange(0,y,1):
        nlattice = lattice[j:j+1,0:x]
        if np.abs(np.mean(nlattice)) !=1:
            a = False 
            j = y
    for i in np.arange(0,x,1):
        nlattice = lattice[0:y,i:i+1]
        if np.abs(np.mean(nlattice)) !=1: 
            b = False
            i = x
    if a == False and b == False:
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
    
def SaveImages3(x,y,p,lattice,z,n):
    if z==0:
        a=display(lattice)
        a.save('outputs2/Image{0}.bmp' .format(int(n)))
    
    
def display(field):
    return Image.fromarray(np.uint8((field)*0.5*500))
    
def InitialLattice(x,y):
    #Generates a lattice of spins. Dimensions x*y
    ilattice = (np.random.randint(2, size=(x,y))*2)-1
    return ilattice

def InitialLatticeDist(x,y,N):
    magV = np.array([])
    for i in np.arange(1,N,1):        
        lattice = InitialLattice(x,y)
        magnetization = TotalMagnitude(lattice)
        magV = np.append(magV, magnetization)
    plt.hist(magV, bins=np.arange(magV.min(), magV.max()+1), orientation='vertical', label = str(x) +"," + str(y) + " N:" + str(x*y))
    plt.legend(loc='upper right')
  
def TotalMagnitudeSq(x):
	magnitudesq = np.sum(x**2)
	return magnitudesq    

    
def TotalMagnitude(x):
    #Summates all the spins to get the total Magnetization of the lattice
    magnitude = np.sum(x)
    return magnitude

def InitialEnergyDist(x,y,N):
    Jd = 0.1
    energyV = np.array([])
    for i in np.arange(1,N,1):        
        lattice = InitialLattice(x,y)
        energy = TotalEnergy(lattice,x,y,Jd)
        energyV = np.append(energyV, energy)
    plt.hist(energyV, bins=np.arange(energyV.min(), energyV.max()+1), orientation='vertical', label = str(x) +"," + str(y) + " N:" + str(x*y))
    plt.legend(loc='upper right')    
      
def TotalEnergy(lattice,x,y,Jd):
    energy = 0
    energysq = 0
    for i in np.arange(0,x,1):
        for j in np.arange(0,y,1):
            currentpoint = lattice[i,j]
            neighbours = lattice[(i+1)%x, j] + lattice[(i-1)%x, j] + lattice[i, (j+1)%y] + lattice[i, (j-1)%y] + Jd*(lattice[(i-1)%x, (j-1)%y] + lattice[(i+1)%x, (j-1)%y] + lattice[(i+1)%x, (j+1)%y] + lattice[(i-1)%x, (j+1)%y])
            energy += -currentpoint*neighbours
            energysq += (currentpoint*neighbours)**2
    return energy/2, energysq/4
    #Gives the energy of a given lattice

    
def Energy(lattice,i,j,x,y,Jd):
    currentpoint = lattice[i,j]
    neighbours = lattice[(i+1)%x, j] + lattice[(i-1)%x, j] + lattice[i, (j+1)%y] + lattice[i, (j-1)%y] + Jd*(lattice[(i-1)%x, (j-1)%y] + lattice[(i+1)%x, (j-1)%y] + lattice[(i+1)%x, (j+1)%y] + lattice[(i-1)%x, (j+1)%y])
    energy = -currentpoint*neighbours
    return energy

def EnergySq (lattice,i,j,x,y,Jd):
    currentpoint = lattice[i,j]
    neighbours = (lattice[(i+1)%x, j])**2 + (lattice[(i-1)%x, j])**2 + (lattice[i, (j+1)%y])**2 + (lattice[i, (j-1)%y])**2 + (Jd*(lattice[(i-1)%x, (j-1)%y]))**2 + (lattice[(i+1)%x, (j-1)%y])** + (lattice[(i+1)%x, (j+1)%y])**2 + (lattice[(i-1)%x, (j+1)%y])**2
    energysq = -currentpoint*neighbours
    return energysq


def Metropolis(lattice,x,y,T,Jd):
    #Applies the Metropolish algorithm
            #Pick a random point in the lattice
    chosenx = np.random.randint(0,x)
    choseny = np.random.randint(0,y)
    iE = Energy(lattice,chosenx,choseny,x,y,Jd)
    if lattice[chosenx,choseny] == 1:
        lattice[chosenx,choseny]=-1
        cE = Energy(lattice,chosenx,choseny,x,y,Jd)
        dE = cE-iE
        if dE>0:
            if np.exp(-((cE-iE)/T)) < random.uniform(0, 1):
                lattice[chosenx,choseny]= 1   
    else:
        lattice[chosenx,choseny]= 1
        cE = Energy(lattice,chosenx,choseny,x,y,Jd)
        dE = cE-iE
        if dE>0:
            if np.exp(-((cE-iE)/T)) < random.uniform(0, 1):
                lattice[chosenx,choseny]= -1
    return lattice

def Imd():
    comp = int(input("How many distributions do you want to compare? "))
    for i in np.arange(0,comp,1):
        print("Choosing Lattice Dimensions")
        x = int(input("x:"))
        y = int(input("y:"))
        InitialLatticeDist(x,y,10000)
    plt.show()

def Ied():
    comp = int(input("How many distributions do you want to compare? "))
    for i in np.arange(0,comp,1):
        print("Choosing Lattice Dimensions")
        x = int(input("x:"))
        y = int(input("y:"))
        InitialEnergyDist(x,y,10000)
    plt.show()


def Alg1():
    T=1
    print("Choosing Lattice Dimensions")
    x = int(input("x:"))
    y = int(input("y:"))
    images = int(input("Do you want to save images in the output folder (Y=1,N=0)"))
    fast = int(input("Fast mode? (doesn't check if finished) (Y=1,N=0)"))
    if fast == 1:
        Alg3(images,x,y)
    elif fast == 0:
        Alg2(images,x,y,T,1)

def Alg2(images,x,y,T,finish,Jd):
    lattice = InitialLattice(x,y)
#    mag = np.array([])
    steps = 25000
    i=0
    p=1000
    full = 0
    TotalMsq=np.array([])
    TotalE=np.array([])
    TotalM=np.array([])
    TotalEsq=np.array([])
    if images == 1:
        n = CheckOutputFiles()
        SaveImages(x,y,0,p,lattice,0,n)
    if finish == 1:
        while CheckForSplit(lattice,x,y) == False:
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            totalMag = TotalMagnitude(lattice)
            TotalM = np.append(TotalM,totalMag)
            totalMagsq = TotalMagnitudeSq(lattice)
            TotalMsq= np.append(TotalMsq,totalMagsq)
            TotalEne, TotalEnesq = TotalEnergy(lattice,x,y,Jd)
            TotalE = np.append(TotalE,TotalEne)
            TotalEsq = np.append(TotalEsq,TotalEnesq)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,i,p,lattice,1,n)
    elif finish == 0:
        for j in np.arange(0,steps,1):
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            totalMag = TotalMagnitude(lattice)
            TotalM = np.append(TotalM,totalMag)
            totalMagsq = TotalMagnitudeSq(lattice)
            TotalMsq = np.append(TotalMsq,totalMagsq)
            TotalEne, TotalEnesq = TotalEnergy(lattice,x,y,Jd)
            TotalE = np.append(TotalE,TotalEne)
            TotalEsq = np.append(TotalEsq,TotalEnesq)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,i,p,lattice,1,n)
    if images == 1:
        SaveImages(x,y,i,p,lattice,1,n)
    if np.abs(TotalMagnitude(lattice)) == x*y:
        full = 1
    avgM = np.mean(TotalM)
    avgMsq = np.mean(TotalMsq)
    avgE = np.mean(TotalE)
    avgEsq = np.mean(TotalEsq)
    # Evar = avgEsq - (avgE*avgE)
	#Mvar = avgMsq - (avgM*avgM)
    return i, avgE, avgEsq, avgM, avgMsq, full#, Evar, Mvar 

def Alg2forsteps(images,x,y,T,finish,Jd):
    lattice = InitialLattice(x,y)
#    mag = np.array([])
    steps = 25000
    i=0
    p=1000
    full = 0
    if images == 1:
        n = CheckOutputFiles()
        SaveImages(x,y,0,p,lattice,0,n)
    if finish == 1:
        while CheckForSplit(lattice,x,y) == False:
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,i,p,lattice,1,n)
    elif finish == 0:
        for j in np.arange(0,steps,1):
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,i,p,lattice,1,n)
    if images == 1:
        SaveImages(x,y,i,p,lattice,1,n)
    if np.abs(TotalMagnitude(lattice)) == x*y:
        full = 1
    # Evar = avgEsq - (avgE*avgE)
	#Mvar = avgMsq - (avgM*avgM)
    return i, full#, Evar, Mvar 

def Alg3(images,x,y):
    lattice = InitialLattice(x,y)
#    mag = np.array([])
    T=3
    l=0
    i=0
    if images == 1:
        p=0
        n = CheckOutputFiles()
        SaveImages2(x,y,p,lattice,0,n)
    while i == 0:
        for a in np.arange(0,x*y,1):
            lattice = Metropolis(lattice,x,y,T)
        l=l+1
        SaveImages2(x,y,l,lattice,1,n)        
    if images == 1:
        SaveImages(x,y,i,p,lattice,1,n)
    return i

def Alg4(images,x,y,T,finish,Jd):
    lattice = InitialLattice(x,y)
#    mag = np.array([])
    steps = 1000
    i=0
    p=5
    full = 0
    if images == 1:
        n = CheckOutputFiles()
        SaveImages(x,y,0,p,lattice,0,n)
    if finish == 1:
        while CheckForSplit(lattice,x,y) == False and i<1000:
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            if images ==1 and i%p == 0: 
                SaveImages(x,y,i,p,lattice,1,n)
    elif finish == 0:
        for j in np.arange(0,steps,1):
            i=i+1
            lattice = Metropolis(lattice,x,y,T,Jd)
            if images ==1 and i%p == 0:
                SaveImages(x,y,i,p,lattice,1,n)
    if images == 1:
        SaveImages(x,y,i,p,lattice,1,n)
    if np.abs(TotalMagnitude(lattice)) == x*y:
        full = 1
   # else:
        #n = CheckOutputFiles2()
        #SaveImages3(x,y,p,lattice,0,n)
    return i, full

def StepsToFinish():
    print(datetime.now())
    T=0.5
    minlattice = 34
    maxlattice = 34
    iterations = 20
    Jd = 0
    fulls=np.array([])
    for j in np.arange(minlattice,maxlattice+1,1): 
        startw = time.time()
        numbersteps= np.array([])
        timea=np.array([])
        ttimea=np.array([])
        x=np.array([])
        for i in np.arange(0,iterations,1):
            start = time.time()
            b, full = Alg2forsteps(0,j,j,T,1,Jd)
            fulls = np.append(fulls, full)
            numbersteps = np.append(numbersteps, b)
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
        print("Total iteration time ", totaltime, " seconds and ", b, "steps. Finish ETA :", finaltime.isoformat(timespec='seconds'))
        SavetoCSV4(j,np.average(numbersteps),np.std(numbersteps),np.average(timea))
        del numbersteps,timea
    print("Percentage full: ", np.sum(fulls)*100/fulls.size, "using ", fulls.size, "simulations")
    print(datetime.now())
    
def ChangeJ():
    print(datetime.now())
    T=0.1
    minJd = 0
    maxJd = 1
    step = 0.1
    iterations = 10000
    for k in np.arange(5,6,1):
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
                b, full= Alg4(0,k,k,T,1,j)
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
                #print(j ,"took " , totaltime , " seconds and ", b, " steps")                  
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
            print("Percentage full: ", percent, "using ", fulls.size, "simulations for ", j, " ", k)
            del numbersteps, timea,fulls
        SavetoCSV3(Jd,percentages,k)
        print(datetime.now())



def Temperature():
    print(datetime.now())
    Jd = 0
    x = 10
    y = 10
    maxT = 1.5
    minT = 0.5
    step = 0.5
    iterations = 15
    for T in np.arange(minT,maxT+step,step): 
        startw = time.time()
        E= np.array([])
        varE= np.array([])
        M= np.array([])
        varM= np.array([])
        latticesizea= np.array([])
        latticesize= np.array([])
        numbersteps= np.array([])
        timea=np.array([])
        ttimea=np.array([])
        temperature=np.array([])
        temperaturea=np.array([])
        averageE=np.array([])
        averageEsq = np.array([])
        Evariance = np.array([])
        averageM=np.array([])
        averageMsq = np.array([])
        Mvariance = np.array([])
        for i in np.arange(0,iterations,1):
            start = time.time()
            latticesizea = np.append(latticesizea, x)
            temperaturea = np.append(temperaturea, T)
            steps, avgE, avgEsq, avgM, avgMsq,full = Alg2(0,x,y,T,0,Jd)
            averageE = np.append(averageE, np.abs(avgE/(x*y)))
            averageEsq = np.append(averageEsq, avgEsq/(x*y))
            Evar = avgEsq/(x*y) - (avgE/(x*y))*(avgE/(x*y))
            Evariance = np.append(Evariance, Evar)
            averageM = np.append(averageM, np.abs(avgM/(x*y)))
            averageMsq = np.append(averageMsq, avgMsq/(x*y))
            Mvar = avgMsq/(x*y) - (avgM/(x*y))*(avgM/(x*y))
            Mvariance = np.append(Mvariance, Mvar)
            numbersteps = np.append(numbersteps, steps)
            end = time.time()
            totaltime = end-start
            timea = np.append(timea,totaltime)
            print(T ,"took " , totaltime , " seconds and ", steps, " steps")
        latticesize = np.append(latticesize, x)
        temperature = np.append(temperature, T)
        E = np.array([np.mean(averageE)])
        varE = np.array([np.std(averageE, ddof=1)])
        M = np.array([np.mean(averageM)])
        varM = np.array([np.std(averageM, ddof=1)])
        Evarmean = np.array([np.mean(Evariance)])
        Evarstd = np.array([np.std(Evariance, ddof=1)])
        Mvarmean = np.array([np.mean(Mvariance)])
        Mvarstd = np.array([np.std(Mvariance, ddof=1)])

    
        #b = plt.scatter(temperature,np.abs(averageE/(x*y)))
        SavetoCSV(latticesizea,numbersteps,timea,averageE,averageEsq,averageM,averageMsq,Evariance,Mvariance,temperaturea)
        SavetoCSV2(latticesize,E,varE,Evarmean,Evarstd,M,varM,Mvarmean,Mvarstd,temperature)
        del latticesize, numbersteps, timea, averageE, averageEsq, averageM, averageMsq, Evariance, Mvariance,temperature,Evarmean,Evarstd,Mvarmean,Mvarstd, latticesizea, temperaturea,E,M,varE,varM
    
        endw = time.time()
        totaltime = (endw-startw)
        ttimea = np.append(ttimea,totaltime)
        totalestimatedtime = (np.mean(ttimea))*(iterations-i-1)
        finaltime = datetime.now()+timedelta(seconds=totalestimatedtime)
        print("Total iteration time ", totaltime, " seconds. Finish ETA :", finaltime.isoformat(timespec='seconds'))
   #b.show()
    print(datetime.now())

def SavetoCSV(lattice,steps,timea,averageE,averageEsq,averageM,averageMsq,Evariance,Mvariance,T):
    df = pd.DataFrame({"latticesize" : lattice, "numbersteps" : steps, "totaltime" : timea, "averageE" :np.abs(averageE), "averageEsq" :np.abs(averageEsq), "averageM" :np.abs(averageM), "averageMsq" :np.abs(averageMsq), "Evariance" : np.abs(Evariance), "Mvariance" :np.abs(Mvariance), "temperature" :T})
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("resultsb11.csv", index=False, mode = 'a')
    
def SavetoCSV2(lattice,E,varE,Evarmean,Evarstd,M,varM,Mvarmean,Mvarstd,temperature):
    df = pd.DataFrame({"latticesize" : lattice, "E" : E, "stdE" : varE, "Evarmean" : np.abs(Evarmean), "Evarstd" : np.abs(Evarstd), "M" :M , "stdM" : varM, "Mvarmean" :np.abs(Mvarmean), "Mvarstd" : np.abs(Mvarstd), "temperature" :temperature})
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("resultsa11.csv", index=False, mode = 'a')    
    
    
def SavetoCSV3(Jd,full,latticesize):
    df = pd.DataFrame({"Jd" : Jd, "percent" : full, "lattice" : latticesize})
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("resultsJd.csv", index=False, mode = 'a')
    
def SavetoCSV4(lattice,numbersteps,stdsteps,timea):
    df = pd.DataFrame({"lattice" : lattice, "steps" : numbersteps, "std steps" : stdsteps,  "time" : timea},  index=[0])
    #df = pd.DataFrame({np.array([lattice,steps,timea,np.abs(averageE),np.abs(averageM),T])})
    df.to_csv("stepsresults.csv", index=False, mode = 'a')
    
    
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
    if choice == 6:
        Imd()
    elif choice == 7:
        Ied()
    elif choice == 1:
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
        
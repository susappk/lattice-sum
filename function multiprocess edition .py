#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:33:20 2017
new editon with function 
@author: teddy
"""
from multiprocessing import Pool
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt("C:/Users/lenovo/Desktop/新建文件夹/Au (Gold) - Palik.txt",unpack='true')
wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt("/Users/zhongguoxia/Desktop/Au (Gold) - Palik.txt",unpack='true')
lam=wavelength_nm
n=fdtd_n+1j*fdtd_k
k_0=2*np.pi/(lam*1e-9)
k_s=k_0
########################################################   
period=57e-9  #period
e=57/6*1e-9  #axis direction initial position distance  
radius=5 #纳米球半径   
con=17*1e-9*3 #六边形边长
wei={}
wei[0]=np.array([np.sqrt(3)*con/2,-con/2,0])
wei[1]=np.array([np.sqrt(3)*con/2,con/2,e])
wei[2]=np.array([0,con,2*e])
wei[3]=np.array([-np.sqrt(3)*con/2,con/2,3*e])
wei[4]=np.array([-np.sqrt(3)*con/2,-con/2,4*e])
wei[5]=np.array([0,-con,5*e])
###########################################################
def j(x):
    return np.sin(x)/x**2-np.cos(x)/x
def jj(x):
    return np.sin(x)+np.cos(x)/x-np.sin(x)/x**2        
def h(x):
    return np.exp(1j*x)*(-1/x-1j/x**2)        
def hh(x):
    return -1j*np.exp(1j*x)+(x+1j)*np.exp(1j*x)/x**2
def alphap(a,k,n):
    rho =k*a*1e-9#半径????       
    a1=(n**2*j(n*rho)*jj(rho)-j(rho)*jj(n*rho))/(n**2*j(n*rho)*hh(rho)-h(rho)*jj(n*rho))
    return 1j*6*np.pi*a1/k**3
#############################################################
def G_axis(r,k_s):
    r=r.reshape(1,3)  
    R=np.linalg.norm(r)        
    I=np.eye(3,dtype=np.complex)
    unittensor=r.T.dot(r)/R**2       
    out=np.zeros((3,3),dtype=np.complex)    
    for i in np.arange(np.size(k_s)):
        A=(1+1j/k_s/R-1/k_s**2/R**2)
        B=(-1-1j*3/k_s/R+3/k_s**2/R**2)
        out=np.exp(1j*k_s*(R-r[0,2]))/R/np.pi/4*(A*I+B*unittensor)
    return out      
def invf_axis (line,lat,k,n):
    I=np.eye(3*line,dtype=np.complex)
    data=np.zeros((lat,3*line,3*line),dtype=np.complex)
    result1=np.zeros((3*line,3*line),dtype=np.complex)
    for m in np.arange(lat):
        f=m-(lat-1)/2            
        #分块填充大矩阵
        for a in np.arange(line):
            for b in np.arange(line):
                if not((f==0) and (a==b)):
                    vector=wei[a]-wei[b]
                    data[m,(0+3*a):(3+3*a),(0+3*b):(3+3*b)]=G_axis((vector+f*period),k)   
        result1=result1+data[m,:,:]
    inverse=np.linalg.inv(I/alphap(radius,k,n)-k**2*result1)
    return inverse
def efield_axis(number,line,k):
    E1=np.zeros((3*line),dtype=np.complex)
    E2=np.zeros((3*line),dtype=np.complex)
    phase=np.zeros((3*line),dtype=np.complex)
        
    E=np.zeros((2,3*line),dtype=np.complex) 
    for a in np.arange(line):
        phase[(0+3*a):(3+3*a)]=np.exp(1j*k*(number*period+np.array([a*e,a*e,a*e])))
        E1[(0+3*a):(3+3*a)]=np.array([1,-1j,0])
        E2[(0+3*a):(3+3*a)]=np.array([1,1j,0])
    E[0,:]=E1*phase/np.linalg.norm(E1*phase)  
    E[1,:]=E2*phase/np.linalg.norm(E2*phase)  
    return E
#######################################################333
def G_crosssection(r,k_s):
    r=r.reshape(1,3)  
    R=np.linalg.norm(r)        
    I=np.eye(3,dtype=np.complex)
    unittensor=r.T.dot(r)/R**2       
    out=np.zeros((3,3),dtype=np.complex)    
    for i in np.arange(np.size(k_s)):
        A=(1+1j/k_s/R-1/k_s**2/R**2)
        B=(-1-1j*3/k_s/R+3/k_s**2/R**2)
        out=np.exp(1j*k_s*(R))/R/np.pi/4*(A*I+B*unittensor)
    return out
def invf_crosssection (line,lat,k,n):
    I=np.eye(3*line,dtype=np.complex)
    data=np.zeros((lat,3*line,3*line),dtype=np.complex)
    result1=np.zeros((3*line,3*line),dtype=np.complex)
    for m in np.arange(lat):
        f=m-(lat-1)/2            
        #分块填充大矩阵
        for a in np.arange(line):
            for b in np.arange(line):
                if not((f==0) and (a==b)):
                    vector=wei[a]-wei[b]
                    data[m,(0+3*a):(3+3*a),(0+3*b):(3+3*b)]=G_crosssection((vector+np.array([0,0,f*period])),k)   
        result1=result1+data[m,:,:]
    inverse=np.linalg.inv(I/alphap(radius,k,n)-k**2*result1)
    return inverse
def efield_crosssection(line,k):
    E1=np.zeros((3*line),dtype=np.complex)
    E2=np.zeros((3*line),dtype=np.complex)
    phase=np.zeros((3*line),dtype=np.complex)   
    E=np.zeros((2,3*line),dtype=np.complex) 
    for a in np.arange(line):
        phase[(0+3*a):(3+3*a)]=np.exp(1j*k*(np.array([wei[a][1]-wei[2][1],wei[a][1]-wei[2][1],wei[a][1]-wei[2][1]])))
        E1[(0+3*a):(3+3*a)]=np.array([1,0,-1j])
        E2[(0+3*a):(3+3*a)]=np.array([1,0,1j])
    E[0,:]=E1*phase/np.linalg.norm(E1*phase)  
    E[1,:]=E2*phase/np.linalg.norm(E2*phase)  
    return E
#########################################################################3
lat=401 
line=6
def f_axis(i):
    print('Run task %s (%s)...' % (i, os.getpid()))
    start = time.time()
    ext=0
    inverse=invf_axis(line,lat,k_0[i],n[i])
    for m in np.arange(lat):
        f=m-(lat-1)/2 
        E=efield_axis(f,line,k_0[i])
        ext=ext+k_0[i]*(E[0,:].conj().dot(inverse.dot(E[0,:]))).imag-k_0[i]*(E[1,:].conj().dot(inverse.dot(E[1,:]))).imag
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (i, (end - start))) 
    return ext

def f_crosssection(i):
    ext2=0
    inverse=invf_crosssection(line,lat,k_0[i],n[i])
    E=efield_crosssection(line,k_0[i])
    ext2=ext2+k_0[i]*(E[0,:].conj().dot(inverse.dot(E[0,:]))).imag-k_0[i]*(E[1,:].conj().dot(inverse.dot(E[1,:]))).imag
    ext2=ext2*401
    return ext2
############################################################################
if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(6)
    result={}
    result2={}
    for i in range(1000):
        result[i]=p.apply_async(f_axis, args=(i,))
    for i in range(1000):
        result2[i]=p.apply_async(f_crosssection, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()    
    p.join()
    print('All subprocesses done.')
##############################################################################    
ext=np.zeros((1000),dtype=np.complex)
ext2=np.zeros((1000),dtype=np.complex)
for i in range(1000):
    ext[i]=result[i].get()
    ext2[i]=result2[i].get()
#########################################################    
print(np.max(ext))
print(np.max(ext2))
plt.figure()
plt.plot(lam,ext,label="$lcp-rcp$")
plt.xlabel("$lambda$")
plt.ylabel("$\sigma_{ext}$")
plt.title("sum")
plt.legend()   

plt.figure()
plt.plot(lam,ext2,label="$lcp-rcp$")
plt.xlabel("$lambda$")
plt.ylabel("$\sigma_{ext}$")
plt.title("sum")
plt.legend()   











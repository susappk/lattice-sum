# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:13:06 2016

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
# 将两列数据分别导入x,y数组
wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt(r"e:\py\Ag (Silver) - Palik (0-2um).txt",unpack='true')
lam=wavelength_nm
n=fdtd_n+1j*fdtd_k
k_0=2*np.pi/(lam*1e-9)
#1st order sphercal bessel function
def j(x):
    return np.sin(x)/x**2-np.cos(x)/x
def jj(x):
    return np.sin(x)+np.cos(x)/x-np.sin(x)/x**2        
def h(x):
    return np.exp(1j*x)*(-1/x-1j/x**2)        
def hh(x):
    return -1j*np.exp(1j*x)+(x+1j)*np.exp(1j*x)/x**2
def alphap(a):
    rho = k_0 * a*1e-9#半径????       
    a1=(n**2*j(n*rho)*jj(rho)-j(rho)*jj(n*rho))/(n**2*j(n*rho)*hh(rho)-h(rho)*jj(n*rho))
    return 1j*6*np.pi*a1/k_0**3
k_s=k_0
#propegater
def G11(R):
    return np.exp(1j*k_s*R)/R/4/np.pi*(1+1j/k_s/R-1/k_s**2/R**2)

frequency_sample=100
radius_sample=50
class alpha:
    size=np.size(n)
    even=np.zeros((radius_sample,frequency_sample,size),dtype=np.complex)
    odd=np.zeros((radius_sample,frequency_sample,size),dtype=np.complex)
    anti=np.zeros((radius_sample,frequency_sample,size),dtype=np.complex)
    same=np.zeros((radius_sample,frequency_sample,size),dtype=np.complex)
    big=np.zeros((radius_sample,frequency_sample),dtype=np.complex)
al=alpha
for l in np.arange(radius_sample):
    for k in np.arange(frequency_sample):
        N=1000#求和项
        point=np.arange(N)
        d=(400+k)*1e-9 #晶格常数      
        E=0;radius=30+l
        for i in point:
            E=E+alphap(radius)*k_0**2*G11((i+1)*d)
        al.same[l,k,:]=alphap(radius)/(1-E)
        al.big[l,k]=np.max(k_0*np.imag(alphap(radius)/(1-E)))/np.pi/(radius*1e-9)**2
#def secondsum(alpha_p1,alpha_p2):
#     for k in np.arange(M):
#        N=500#求和项
#        point=np.arange(N)
#        d=(380+30*k)*1e-9 #晶格常数
#        A=0;B=0;C=0;D=0;
#        for i in point:
#            A=A+alpha_p1*k_0**2*G11(2*(i+1)*d)#偶数个对零点求和，零点除外
#            B=B+alpha_p1*k_0**2*G11((2*i+1)*d)#奇数个对零点求和
#            C=C+alpha_p2*k_0**2*G11((2*i+1)*d)#偶数个对一点求和
#            D=D+alpha_p2*k_0**2*G11(2*(i+1)*d)#奇数个对一点求和，一点除外
#        al.even[k,:]=(alpha_p1*(1-D)+B*alpha_p2)/((1-A)*(1-D)-B*C)
#        al.odd[k,:]=(alpha_p2*(1-A)+C*alpha_p1)/((1-A)*(1-D)-B*C)
#    al.anti=al.even+al.odd
#画出ext--frequency ,radius 条曲线
plt.figure()
for l in np.arange(radius_sample):
    plt.plot(np.arange(frequency_sample)+400,al.big[l,:])
plt.xlabel("$frequecy$")
#plt.xlim([300,600])
plt.ylabel("$\sigma_{ext}$ max")
plt.title("ext--frequency with radius varing from 10-(+5)-80")
plt.legend()
plt.show()
#画出ext--radius 
b=np.zeros(radius_sample,dtype=np.complex)
for l in np.arange(radius_sample):
    b[l]=np.max(al.big[l,:])
plt.figure()
plt.plot(np.arange(radius_sample)+30,b,label="$1st$")
plt.xlabel("$radius$")
#plt.xlim([300,600])
plt.ylabel("$\sigma_{ext}$ max")
plt.title("radius vs max extinction cross section peak")
plt.legend()
plt.show()

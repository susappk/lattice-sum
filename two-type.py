import numpy as np
import matplotlib.pyplot as plt
# 将两列数据分别导入x,y数组
wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt(r"e:\py\Ag (Silver) - Palik (0-2um).txt",unpack='true')
lam=wavelength_nm
n=fdtd_n+fdtd_k
k_0=(2*np.pi/(lam*1e-9))
#1st order sphercal bessel function
def j(x):
    return np.sin(x)/x**2-np.cos(x)/x
def jj(x):
    return np.sin(x)+np.cos(x)/x-np.sin(x)/x**2        
def h(x):
    return np.exp(1j*x)*(-1/x-1j/x**2)        
def hh(x):
    return -1j*np.exp(1j*x)+(x+1j)*np.exp(1j*x)/x**2
def a1(a):
    rho = k_0 * a*1e-9#半径????       
    return (n**2*j(n*rho)*jj(rho)-j(rho)*jj(n*rho))/(n**2*j(n*rho)*hh(rho)-h(rho)*jj(n*rho))

k_s=k_0
#propegater

#求和x**2/R**2-1j*3*x**2/k_s/R**3+3*x**2/k_s**2/R**4
same=np.zeros((1000),dtype=np.complex)
N=3000#求和项



def G11(R):
    return np.exp(1j*k_s*R)/R/4*(1+1j/k_s/R-1/k_s**2/R**2)
point=np.arange(N)
d=(480)*1e-9 #晶格常数
E=0
point=np.arange(2*N)
for i in point:
    E=E+G11((i+1)*d)
a=(k_0**2/(1j*6*a1(50)))
b=k_0*E

c=np.real(a)
d=np.imag(a)
e=np.real(b)
f=np.imag(b)
g=(c-e)
k=d-f
#ext=1/(c+1j*d-e-1j*f)
ext=-k/(g**2+k**2)



plt.figure(1)
plt.plot(lam,ext)
plt.xlabel("$\lambda$")
plt.xlim([430,520])
#plt.ylim([-1.1,1.1])
plt.ylabel("$\sigma_{ext}$")
plt.title("ext")
plt.legend()
plt.show()



plt.figure(2)
plt.plot(lam,c,label="c")
plt.plot(lam,d,label="d")
plt.xlabel("$\lambda$")
plt.xlim([430,520])
plt.ylabel("$\sigma_{ext}$")
plt.title("ext")
plt.legend()
plt.show()

plt.figure(3)
plt.plot(lam,e,label="e")
plt.plot(lam,f,label="f")
plt.xlabel("$\lambda$")
plt.xlim([430,520])
plt.ylabel("$\sigma_{ext}$")
plt.title("ext")
plt.legend()
plt.show()

plt.figure(4)
plt.plot(lam,g,label="g")
plt.plot(lam,k,label="k")
plt.xlabel("$\lambda$")
plt.xlim([430,520])
plt.ylim([-0.3e15,0.3e15])
plt.ylabel("$\sigma_{ext}$")
plt.title("ext")
plt.legend()
plt.show()

plt.figure(4)
plt.plot(lam,f/(e**2+f**2),label="g")

plt.xlabel("$\lambda$")
plt.xlim([430,520])
plt.ylabel("$\sigma_{ext}$")
plt.title("ext")
plt.legend()
plt.show()
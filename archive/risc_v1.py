import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, SpatialDim = 1, nx = 0, ny=0, nz=0, xrange=[0,1], yrange=[0,1], zrange = [0,1]
                 , trange=[0,1], sigma = 0, BC= "", IC=""):
        super().__init__()
        self.Dim = SpatialDim
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.trange = trange

        self.sigma = sigma

        self.BC = BC
        self.IC = IC

        self.Matrix = np.zeros([0])

        self.solved = False

        self.solverType = ""
        self.paramtersSet = False

        self.dx = (self.xrange[1] - self.xrange[0])/ (self.nx - 1)
        #self.dy = (self.yrange[1] - self.yrange[0])/ (self.ny - 1)
        #self.dz = (self.zrange[1] - self.zrange[0])/ (self.nz - 1)


        self.dt = self.sigma * self.dx * self.dx
        self.nt = int(self.trange[1] / self.dt)


        self.x = np.linspace(self.xrange[0],self.xrange[1],self.nx)
        self.y = np.linspace(self.yrange[0],self.yrange[1],self.ny)
        self.z = np.linspace(self.zrange[0],self.zrange[1],self.nz)

    @abstractmethod
    def setBoundaryConditions(self, BC):
        self.BC = BC

    @abstractmethod
    def setInitialConditions(self, IC):
        self.IC = IC

    @abstractmethod
    def setModelParameters(self):
        raise NotImplementedError("Cannot Set Parameters from Abstract Class")

    @abstractmethod
    def runModel(self):
        raise NotImplementedError("Cannot Solve from Abstract Class")

    @abstractmethod
    def getSolvedValues(self):
        raise NotImplementedError("Cannot retrieve solved values for Abstract Class")

    @abstractmethod
    def Plot(self):
        raise NotImplementedError("Cannot Plot from Abstract Class")
    


class RISC_Model_1d(BaseModel):

    def __init__(self, SpatialDim, nx, xrange, trange, sigma, BC, IC):
        super().__init__(SpatialDim = SpatialDim, nx = nx,
                         xrange=xrange,  trange=trange, 
                         sigma=sigma, BC=BC, IC=IC)
        self.beta = 0
        self.gamma = 0
        self.alpha_i = 0
        self.alpha_s = 0
        self.alpha_c = 0
        self.N_i = 0
        self.N_s = 0
        self.N_c = 0
        self.dr = 0
        self.di = 0
        self.ds = 0
        self.dc = 0
        self.paramtersSet = False

        self.dx = (self.xrange[1] - self.xrange[0])/ (self.nx - 1)
        nx = self.nx



        self.dt = self.sigma * self.dx * self.dx
        self.nt = int(self.trange[1] / self.dt)



        self.t = np.linspace(trange[0], trange[1], self.nt)



        self.r = np.ones((nx)) * 0.1
        self.i = np.ones((nx)) * 1
        self.s = np.ones((nx)) * 0
        self.c = np.ones((nx)) * .05

        rng = np.random.default_rng()
        self.c = rng.normal(loc=0.0, scale=.1, size=self.c.shape) + .2



        self.rn = np.ones((nx))
        self.In = np.ones((nx))
        self.sn = np.ones((nx))
        self.cn = np.ones((nx))

        # initial conditions
        self.r[int(1/2 * nx):-1] = 0
        self.i[int(1/2 * nx):-1] = 0
        self.s[0:int(1/2 * nx)] = 0
        self.c = 5 * self.x * ( 1 - self.x)



        self.c0 = self.c.copy()
        # store all solutions
        self.tmp = np.zeros([0])
        self.tmp = np.append(self.tmp, self.c[int(nx/2)])

        self.RSolutionMatrix = np.zeros((self.nt,self.nx))
        self.RSolutionMatrix[0,:] = self.r

        self.ISolutionMatrix = np.zeros((self.nt,self.nx))
        self.ISolutionMatrix[0,:] = self.i

        self.SSolutionMatrix = np.zeros((self.nt,self.nx))
        self.SSolutionMatrix[0,:] = self.s

        self.CSolutionMatrix = np.zeros((self.nt,self.nx))
        self.CSolutionMatrix[0,:] = self.c


    def setModelParameters(self, beta, gamma, alpha_i, alpha_s, alpha_c, N_i, N_s, N_c, dr, di, ds, dc):
        self.beta = beta
        self.gamma = gamma
        self.alpha_i = alpha_i
        self.alpha_s = alpha_s
        self.alpha_c = alpha_c
        self.N_i = N_i
        self.N_s = N_s
        self.N_c = N_c
        self.dr = dr
        self.di = di
        self.ds = ds
        self.dc = dc
        self.paramtersSet = True

    def runModel(self):
        for n in range(self.nt): ##loop across number of time steps

            self.rn = self.r.copy() ; 
            self.In = self.i.copy()
            self.sn = self.s.copy()
            self.cn = self.c.copy()

            self.r[1:-1] = self.rn[1:-1]  + self.dt * ( self.gamma - self.beta * self.rn[1:-1]*self.In[1:-1] ) + \
                self.dt / self.dx**2 * self.dr * (self.rn[2:] - 2 * self.rn[1:-1] + self.rn[:-2])

            self.i[1:-1] = self.In[1:-1]  + self.dt * ( ( self.alpha_i * self.In[1:-1] * (1.0 - self.In[1:-1] / self.N_i) - \
                self.beta * self.rn[1:-1]*self.In[1:-1])  ) + \
                self.dt / self.dx**2 * self.di * (self.In[2:] - 2 * self.In[1:-1] + self.In[:-2])

            self.s[1:-1] = self.sn[1:-1]  + self.dt*( ( self.beta * self.rn[1:-1] * self.In[1:-1])  + \
                self.alpha_i * self.sn[1:-1] * (1.0 - self.sn[1:-1] / self.N_s) ) + \
                self.dt / self.dx**2 * self.ds * (self.sn[2:] - 2 * self.sn[1:-1] + self.sn[:-2])

            self.c[1:-1] = self.cn[1:-1] +  self.dt * ( self.alpha_c * self.cn[1:-1] * (1.0 - self.cn[1:-1] / self.N_c)  - \
                self.sn[1:-1] * self.cn[1:-1]) + \
                self.dt / self.dx**2 * self.dc * (self.cn[2:] - 2 * self.cn[1:-1] + self.cn[:-2])


            self.r[0] = self.rn[0] + self.dt / self.dx**2 * 2 * (self.rn[1] - self.rn[0])
            self.r[-1] = self.rn[-1] + self.dt / self.dx**2 * 2 * (self.rn[-2] - self.rn[-1])

            self.i[0] = self.In[0] + self.dt / self.dx**2 * 2 * (self.In[1] - self.In[0])
            self.i[-1] = self.In[-1] + self.dt / self.dx**2 * 2 * (self.In[-2] - self.In[-1])

            self.s[0] = self.sn[0] + self.dt / self.dx**2 * 2 * (self.sn[1] - self.sn[0])
            self.s[-1] = self.sn[-1] + self.dt / self.dx**2 * 2 * (self.sn[-2] - self.sn[-1])

            self.c[0] = self.cn[0] + self.dt / self.dx**2 * 2 * (self.cn[1] - self.cn[0])
            self.c[-1] = self.cn[-1] + self.dt / self.dx**2 * 2 * (self.cn[-2] - self.cn[-1])
            self.tmp = np.append(self.tmp, self.c[int(self.nx/2)])

            self.RSolutionMatrix[n,:] = self.r
            self.ISolutionMatrix[n,:] = self.i
            self.SSolutionMatrix[n,:] = self.s
            self.CSolutionMatrix[n,:] = self.c



        T,X = np.meshgrid(self.t, self.x)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,  figsize=(12,7), subplot_kw={'projection': '3d'})

        # 3. Plot the first surface on ax1
        surf1 = ax1.plot_surface(X,T, self.RSolutionMatrix.T, cmap='inferno') #, edgecolor='none')
        ax1.set_title("r")
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$t$')

        surf2 = ax2.plot_surface(X,T, self.ISolutionMatrix.T, cmap='inferno', edgecolor='none')
        ax2.set_title("i")
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$t$')

        surf3 = ax3.plot_surface(X,T, self.SSolutionMatrix.T, cmap='inferno', edgecolor='none')
        ax3.set_title("s")
        ax3.set_xlabel('$x$')
        ax3.set_ylabel('$t$')

        surf4 = ax4.plot_surface(X,T, self.CSolutionMatrix.T, cmap='inferno', edgecolor='none')
        ax4.set_title("c")
        ax4.set_xlabel('$x$')
        ax4.set_ylabel('$t$')


        plt.tight_layout()
        plt.show()

         

    def setBoundaryConditions(self, BC):
        self.BC = BC


    def setInitialConditions(self, IC):
        self.IC = IC



    def getSolvedValues(self):
        pass
  


    def Plot(self):
        pass


    





 
if __name__ == "__main__":
    print("Computing RISC Model")
    risc = RISC_Model_1d(SpatialDim=1, nx = 32, xrange = [0,1],trange = [0, 3.5], sigma=.3, BC='Neumann', IC= "basic")
    risc.setBoundaryConditions(BC = 'Neumann')
    risc.setModelParameters(beta = 2, gamma = 1.5, alpha_i = 4.4, alpha_s = .3, alpha_c = 4.8, N_i = 4, N_s = 4, N_c = 4,
                            dr = .1, di=.3, ds = .1, dc=.03)
    risc.runModel()
    
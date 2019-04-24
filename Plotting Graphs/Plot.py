'''
--  Mathematical graphing capabilities of Python --

`helper`
    -- plot functions
        - A set of helper functions

`examples`
    - steps
        - Approximating the step function

    - goldenRatio
        - Understaning the origin on the fibonacci numbers

    - aperyConstant
        - Aproximating the Apéry's Constant

    - eulerConstant
        - Aproximating the Euler's Constant

`functions`
    - linear function

    - quadratic function

    - parabola

    - hyperbola

    - exponential function

    - logarithmic function

    - circle
        - Plotting a circle

`trig`
    - Trigonometry:
        - sin
        - cos
        - tg
        - ctg
        - arctg
        - arcsin
        - arccos

`threeD`
    - surface
        - Graphing a 3d surface

    - TD_king
        - A nice example of a 3d surface

    - cubeVol
        - Representing a cube as a set of 3D points

    - plotPyramidSurf
        - Plotting a pyramid


`solver`
    - solve
        -Solving a set of equations

    - solveVisual
        -Solving a set of mutually equal equations visually

`series`
    - powerSeries
        - Drawing parametrized power series and their sum

'''

from math import log as m_log
from math import atan, gcd
from random import randrange

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sympy import solve as solve_eq
from sympy import Eq, sympify, var
from sympy.plotting import plot as plotSym

details=10  # how small are the steps on the plot
scale=100.0  # x ranges between 0 and scale / or -scale to scale

class helper:
    
    def plot(self, x,y, name=""):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.plot(x, y)
        plt.title(name)
        plt.show()

    def plotMultiple(self, arr, name=""):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.title(name)

        for cords in arr:
            plt.plot(cords[0], label=cords[1])

        plt.legend()
        plt.show()
        


    def plotPow(self, x,y, name=""):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.plot(x, y)
        plt.ylim((-scale/4, scale/4))
        plt.xlim((-scale/4, scale/4))
        plt.title(name)
        plt.show()
        plt.ylim(auto=True)
        plt.xlim(auto=True)


    def plotExp(self, x,y, name=""):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.plot(x, y)
        plt.ylim(-1, 8)
        plt.xlim(-1, 8)
        plt.title(name)
        plt.show()
        plt.ylim(auto=True)
        plt.xlim(auto=True)


    def plotTrig(self, x,y, name=""):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.plot(x, y)
        plt.ylim((-2, 2))
        plt.xlim((-10, 10))
        plt.title(name)
        plt.show()
        plt.ylim(auto=True)
        plt.xlim(auto=True)


    def plot3D(self, x, y, z, name=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=2, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(name)
        plt.show()


    def plot3D_line(self, x, y, z, name=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z)
        ax.legend()
        plt.title(name)
        plt.show()


    def plotCube(self, x, y, z,name=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.legend()
        ax.scatter(x,y,z)
        plt.title(name)
        plt.show()

    def plotEuler(self, x,y, name="-e-"):
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axhline(1, color='red', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.plot(x, y)
        plt.ylim(-1, 8)
        plt.xlim(-1, 8)
        plt.title(name)
        plt.show()
        plt.ylim(auto=True)
        plt.xlim(auto=True)

class examples(helper):

    def steps(self):
        '''
        Approximate the 'Heaviside Step Function' of increasing 'steps'
        (http://mathworld.wolfram.com/HeavisideStepFunction.html)
        '''
        
        tune=5       # must be 2,4,6,8,etc. The greater the better

        x = np.arange(0.0, scale, 1/details)
        y = np.arange(0.0, scale, 1/details)

        #base case
        for i in range(0,details):
            y[i]=1


        for i in range(details,int(scale)*details):
            y[i]= y[i-1]+1/np.power((i%details-0.99),tune*2)/np.power(scale,tune)

        self.plot(x,y)


    def goldenRatio(self, a=1,b=1):
        '''
        Getting the Golden Ratio with 'Brady numbers'
        (http://www.numberphile.com/videos/brady_numbers.html)
        '''

        x = np.arange(0.0, scale, 1/details)
        y = np.arange(0.0, scale, 1/details)
        z = np.arange(0.0, scale, 1/details)

        #base case
        y[0]=a
        y[1]=b


        for i in range(2,int(scale)*details):
            y[i]= y[i-1]+y[i-2]

        for i in range(2,int(scale)*details):
            z[i]= y[i]/y[i-1]

        self.plotPow(x,z, "The golden ratio")


    def aperyConstant(self, n=10000, max=10):
        '''
        Getting the Apéry's Constant as demonstrated here:
        https://www.youtube.com/watch?v=ur-iLy4z3QE

        (http://mathworld.wolfram.com/AperysConstant.html)
        '''
        
        x = np.arange(0.0, n, 1/details)
        y = np.arange(0.0, n, 1/details, dtype=object)
        z = np.arange(0.0, n, 1/details)

        for i in range(0,int(n)*details):
            y[i]= np.array([randrange(max),randrange(max),randrange(max)])

        num=0
        for i in range(0,int(n)*details):
            if( gcd( gcd(y[i][0],y[i][1]), y[i][2])  == 1 ):
                num+=1
            if num>0:
                z[i]= i/num

        self.plot(x,z)


    def eulerConstant(self, dx=0.0001, n=10):
        '''
        - Finding a number 'e' such that (e^dx-1)/dx = 1:
        https://www.youtube.com/watch?v=m2MIpDrF7Es

        - Finding 'e' by seeing what the function (1+1/n)^n approaches:
        http://www.mathsisfun.com/numbers/e-eulers-number.html

        (http://mathworld.wolfram.com/e.html)
        '''
        
        x = np.arange(0.0, n*100, 1/details)
        y = (np.power(x, dx) - 1) / dx

        self.plotEuler(x,y)

        y = np.power( 1 + 1/x, x )
        self.plot(x,y)

class functions(helper):

    def linear(self, k=1,c=0):
        x = np.arange(0, scale) 
        y= k*x+c

        self.plot(x,y)


    def quadratic(self, a=1, b=1, c=0):
        x = np.arange(-scale, scale, 1/details)
        y= a*np.power(x,2)+b*x+c

        self.plotPow(x,y)
        

    def parabola(self, selfpow, a=0,k=1,b=0):
        x = np.arange(-scale, scale, 1/details)
        y= k*np.power(x+a,pow)+b

        self.plotPow(x,y)


    def hyperbola(self, pow, a=0, k=1, b=0):
        x = np.arange(-scale, scale, 1/details)
        y= k*(1/np.power(x+a,pow))+b

        self.plotPow(x,y)


    def exponential(self, base,k=1.0,b=0):
        x = np.arange(0, 10, 1/details)
        y = np.arange(0, 10, 1/details)

        for i in range(0,10*details):
            y[i]= k*np.power(base,i/details)+b
            if np.isnan(y[i]):
                y[i]=0 

        self.plotExp(x,y)


    def log(self, base,k=1.0,b=0):
        x = np.arange(0, 10, 1/details)
        y = np.arange(0, 10, 1/details)

        if base>0 and base<1:
            y[0]=100
        elif base>=1:
            y[0]=-100
        else:
            return 0
            
        for i in range(1,10*details):
            y[i]= k*m_log(i/details,base)+b
            if np.isnan(y[i]):
                y[i]=0 

        self.plotExp(x,y)

    
    def circle(self):
        x = np.arange(-scale, scale, 1/details)

        y=np.sin(x)
        x=np.cos(x)

        self.plot(x,y)

    class trig(helper):
        
        def sin(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*np.sin(k*x + w) + b

            self.plotTrig(x,y)


        def cos(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*np.cos(k*x + w) + b

            self.plotTrig(x,y)


        def tg(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*np.tan(k*x + w) + b

            self.plotTrig(x,y)


        def ctg(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a/np.tan(k*x + w) + b

            self.plotTrig(x,y)


        def arctg(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*atan(k*x + w) + b

            self.plotTrig(x,y)


        def arcsin(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*np.arcsin(k*x + w) + b

            self.plotTrig(x,y)


        def arccos(self, a=1,b=0,w=0,k=1):
            x = np.arange(-scale, scale, 1/details)
            y= a*np.arccos(k*x + w) + b

            self.plotTrig(x,y)


        def cot(self, x):
            if x==0:
                return x
            x = np.divide(np.cos(x), np.sin(x))
            return x

class threeD(helper):
    
    def surface(self, h=2,k=3,a=3,b=2,c=6):
        x = np.arange(-scale, scale, 1/details)
        y = np.arange(-scale, scale, 1/details)
        x, y = np.meshgrid(x, y)
        z = np.arange(-scale, scale, 1/details)

        z=h*x**a+k*y**b+c

        self.plot3D(x,y,z)


    def cubeVol(self, a=1):
        x,y,z = np.mgrid[0:a:1/details, 0:a:1/details,  0:a:1/details]
        self.plotCube(x,y,z)


    def TD_king(self):
        x = np.arange(-scale, scale, 1/details)
        y = np.arange(-scale, scale, 1/details)
        x, y = np.meshgrid(x, y)
        z = np.arange(-scale, scale, 1/details)

        z=(x**3/np.e-y**3/np.e)/(9*np.e**6)

        self.plot3D(x,y,z)


    def three_D_line(self):
        x = np.arange(-scale, scale, 1/details)
        y = np.arange(-scale, scale, 1/details)
        z = np.arange(-scale, scale, 1/details)

        for i in range(0,int(scale)*details):
            y[i]=x[i]**2
            z[i]=y[i]**2

        self.plot3D_line(x,y,z)

    def plotPyramidSurf(self, v=np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1],  [-1, 1, -1], [0, 0, 1]]),name=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
        # generate list of sides' polygons of our pyramid
        verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
        [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, 
        facecolors='y', linewidths=.5, edgecolors='gray'))
        plt.title(name)
        plt.show()

class solver:
        
    def solve(self, eqs=[("x^2+2*x-4","0")]): 
        eqs = [Eq(sympify(eq),sympify(sol)) for eq, sol in eqs]
        print(solve_eq(eqs))

        #solve([("x^2+y^2","100"),("x^2-y","28")])

    def solveVisual(self, eqs=["x^2+2*x", "2*x+9"]):
        eqs = [sympify(eq) for eq in eqs]
        eqsPlot=plotSym(eqs[0], show=False)

        for i in range(1,len(eqs)):
            eqsPlot.extend(plotSym(eqs[i], show=False))

        eqsPlot.show()


class series(helper):

    def powerSeries(self, p=2, c=1):
        # An = c*(1/n)^p
        x = np.arange(1, scale, 1)
        y = c*np.power(x,-p)

        # Sn = A1+A2...An
        s = [ np.sum( y[:n] ) for n in range(y.shape[0]) ]

        self.plotMultiple([[y, "An"], [s, "Sn"]], str(c)[:5]+"* (1/n)^"+str(p)[:5])

 
s = series()
s.powerSeries(2)

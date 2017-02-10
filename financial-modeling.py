#!/usr/bin/env python

import numpy as np


class InterestRate:
    """
    This class represents the interest rate,
    or the value of a bank account/bond.
    """
    
    def __init__(self, muB=0.05, sigmaB=0.05, B0=1):
        """
        Initialize the InterestRate objects. 
        muB is the drift, sigmaB is the volatility, B0
        is the start value. 
        """
        
        self.muB = muB
        self.sigmaB = sigmaB
        self.B0 = B0


    def __repr__(self):
        """
        String representation of an instance, for
        recreation of the instance.
        """
        
        return "InterestRate(" + str(self.muB) + ", " + str(self.sigmaB) +", "+ str(self.B0) + ")"

    
    def calculate(self, T, epsilon, n=10000):
        """
        This function takes the number of timesteps T, a Txn matrix of
        N(0,1) distributed variables epsilon and the number of
        Monte Carlo simulations n as arguments. Then it calculates
        and returns the interest rate/bond values B.
        """
        
        B = np.zeros(n*T) + self.B0
        B.shape = (T, n)
                
        for i in range(T):
            B[i, ] = self.B0*np.exp((self.muB-self.sigmaB**2/2)*i + self.sigmaB*epsilon[i,])

        return B



class GBM:
    """
    This class represents stocks modelled via geometric
    brownian motion.
    """

    def __init__(self, muS=0.10, sigmaS=0.20, S0=1):
        """
        Initialize the parameters in the GBM model.
        muS is the drift and sigmaS is the volatility.
        S0 is the stock's start value. 
        """
        
        self.muS = muS
        self.sigmaS = sigmaS
        self.S0 = S0


    def __repr__(self):
        """
        String representation of an instance of the class, for
        recreation of the instance.
        """
        
        return "GBM(" + str(self.muS) +", "+ str(self.sigmaS) +", "+ str(self.S0) +")"

        
    def calculate(self, T, epsilon, n=10000):
        """
        This function takes the number of timesteps T, a Txn matrix of
        N(0,1) distributed variables epsilon and the number of
        Monte Carlo simulations n as arguments. Then it calculates
        and returns the stock values S.
        """

        S = np.zeros((n*T))
        S.shape=(T, n)
        
        for i in range(T):
            S[i, ] = self.S0*np.exp((self.muS-self.sigmaS**2/2)*i + self.sigmaS*epsilon[i,])          
            
        return S



class GARCH:
    """
    This class represents the GARCH(1,1) model for stock prices.
    """
        
    def __init__(self, S0=1, mu=0, theta0=0.000002, theta1=0.09, theta2=0.89, k=3):
        """
        This function initialize the GARCH-objects.
        It also calculates the start-values of the sigma-function,
        the stochastic standard deviation.
        """
        
        self.S0 = S0
        self.mu = mu
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.k = k
        self.sigma0 = k * np.sqrt(theta0/(1-(theta1+theta2)))
        self.nu = (self.mu - self.sigma0**2/2)


    def __repr__(self):
        """
        String representation of an instance, for
        recreation of the instance.
        """

        return "GARCH(" +str(self.S0) + ", "+ str(self.mu) +", "+ str(self.theta0) +", "+ str(self.theta1) +", "+ str(self.theta2) +", "+ str(self.k) +")"


    def calculate(self, T, epsilon, n=10000):
        """
        This function takes the number of timesteps T, a matrix of
        N(0,1) distributed variables epsilon and the number of
        Monte Carlo simulations n as arguments. Then it calculates
        and returns the stock values S.
        """
        
        sigma = np.zeros(n*T) + self.sigma0
        sigma.shape = (T, n)
        S = np.zeros(n*T) + self.S0
        S.shape = (T, n)
        
        
        for i in range(T):
            sigma[i, 1:n-1] = np.sqrt(self.theta0 + self.theta1*(sigma[i, 0:n-2]*epsilon[i, 0:n-2])**2 + self.theta2*(sigma[i, 0:n-2]))
            
            S[i, 1:n] = S[i, 0:n-1]*np.exp(self.mu + sigma[i, 1:n]*epsilon[i, 0:n-1])
        
        return S, sigma




class Portfolio:
    """
    This class represents a portfolio with a bond and a stock,
    where the bond and stock follows geometric brownian motion.
    """
    
    def __init__(self, alpha, rho=0, muB=0.05, sigmaB=0.05, B0=1, muS=0.10, sigmaS=0.2, S0=1):
        """
        This function initialize the Portfolio-object. 
        Alpha is the amount of stocks in the portfolio, rho is the correlation between
        the stock and the bond values, the mus are the drift and the sigmas
        are the volatility of the bond and stock, B0 and S0 is the start values.
        """

        self.alpha = alpha
        self.rho = rho
        self.muB = muB
        self.sigmaB = sigmaB
        self.B0 = B0
        self.muS = muS
        self.sigmaS = sigmaS
        self.S0 = S0
        

    def __repr__(self):
        """
        String representation of the Portfolio class,
        which can be used to recreate the object.
        """

        return "Portfolio(" +str(self.alpha) +", "+ str(self.rho) +", "+ str(self.muB) +", "+ str(self.sigmaB) +", "+ str(self.B0) +", "+str(self.muS) +", "+ str(self.sigmaS) +", "+ str(self.S0) +")"
               
    

    def calculate(self, T, n=10000):
        """
        This function takes the number of timesteps T and the
        number of Monte Carlo-simulations n as arguments.
        It calculates and returns the value of the defined portfolio,
        using the calculate-functions of the GBM and InterestRate
        classes.
        """
        
        epsilon1, epsilon2 = calcCorrNormal(T, self.rho, n)
        b = InterestRate(self.muB, self.sigmaB, self.B0)
        bond = b.calculate(T, epsilon1, n)
        s = GBM(self.muS, self.sigmaS, self.S0)
        stock = s.calculate(T, epsilon2, n)
        portfolio = self.alpha*stock + (1-self.alpha)*bond
        return portfolio
        


    


def calcCorrNormal(T, rho=0.6, n=10000):
    """
    This function takes the number of timesteps T,
    the correlation rho and the number of Monte Carlo
    simulations n as arguments. It calculates and returns
    two rho-correlated Txn-matrices epsilon1 and epsilon2
    from the N(0,1)-distribution.
    """

    epsilon1 = np.random.standard_normal(n*T)
    epsilon3 = np.random.standard_normal(n*T)
    epsilon2 = rho*epsilon1 + epsilon3*np.sqrt(1-rho**2)
    
    epsilon1.shape = (T, n)
    epsilon2.shape = (T, n)

    return epsilon1, epsilon2



def calcUncorrNormal(T, n=10000):
    """
    This function takes the number of timesteps T and
    the number of Monte Carlo simulations as arguments.
    It creates and returns a Txn-matrix of independent
    Monte Carlo- realizations from the N(0,1) distribution.
    """
    
    epsilon = np.random.standard_normal(n*T)
    epsilon.shape = (T, n)

    return epsilon
    

def sortMC(X):
    """
    This function takes a matrix X as argument, 
    and sorts it (for instance, each timestep),
    individually. It returns the sorted matrix. 
    """

    if X.shape[0]<X.shape[1]:
        #Row-major storage 
        for i in xrange(X.shape[0]):
            X[i, :] = np.sort(X[i, :])
    else:
        #Column-major storage
        for i in xrange(X.shape[1]):
            X[:,i] = np.sort(X[:, i])
        
    return X
    

    
def calcQuantile(q, X, sorted=False):
    """
    This function calculates and returns the
    approximate q-quantile for each time-step t=1,...,T
    of a Txn-matrix X. For a sorted sequence
    of m Monte Carlo simulations X(1)<=X(2)<=...<=X(m),
    the approximate e-quantile is given as X(m*e).
    """
    
    if not sorted:
        X = sortMC(X)
            
    quantiles = np.zeros(X.shape[0])
    
    #Calculates the approximate q-quantile for each timestep   
        
    quantiles[0:X.shape[0]] = X[0:X.shape[1], int(q*X.shape[1])] 
    return quantiles



def calcMean(X):
    """
    This function calculates and returns the
    sample mean (average value) at each timestep
    t=1,...,T of a Txn-matrix X.
    """
    
    if X.shape[0]<X.shape[1]:
        #Row-major storage
        
        xMean = np.zeros(X.shape[0])
        
        for i in xrange(X.shape[0]):
            xMean[i] = sum(X[i, :])
        return xMean / X.shape[1]
    else:
        xMean = zeros(X.shape[1], Float)

        for i in xrange(X.shape[1], Float):
            xMean[i] = sum(X[:, i])
        
        return xMean  / X.shape[0]

        

def calcStDev(X):
    """
    This function calculates and returns the
    sample standard deviation at each timestep
    t=1,...,T of a Txn-matrix X. 
    """
    
    stDev = np.zeros(X.shape[1])
    mean = calcMean(X)
        
    #Calculates the sample standard deviation
    #for each timestep:
    
    #for i in range(len(stDev)):
        #stDev[i] = np.sqrt((sum(X[i,] - mean[i])**2)/(X.shape[1]-1))
        #<<TBC>>
    return stDev



if __name__ == "__main__":    
    """ Example of financial analysis """

    alpha = 0.20  #The proportion of stocks in the portfolio
    rho = 0.6  #The correlation between stocks and bonds
    T = 120  #The number of timesteps
    n = 1000  #The number of Monte Carlo simulations

    s1 = GBM()
    s2 = GARCH()
    epsilon1, epsilon2 = calcCorrNormal(T, rho, n)
    s1Sim = s1.calculate(T, epsilon1, n)
    s2Sim, sigma = s2.calculate(T, epsilon2, n)
    
    p = Portfolio(alpha, rho)
    portfolio = p.calculate(T, n) #The calculated portfolio values
    median = calcQuantile(0.5, portfolio, sorted=False) #The medians
    upper1 = calcQuantile(0.99, portfolio, sorted=False) #The upper 1% quantiles
    lower1 = calcQuantile(0.01, portfolio, sorted=False) #The lower 1% quantiles
    mean = calcMean(portfolio) #The mean observations
    stdev = calcStDev(portfolio) #The standard deviations
    
    #print "The means: " + str(mean)
    #print "The std deviations: " + str(stdev)
    #print "The medians: " + str(median)
    #print "The 1% quantiles: " + str(lower1)
    #print "The 99% quantiles: " + str(upper1)

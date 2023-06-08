from tkinter import N
import numpy as np
import itertools
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import csv

import pandas as pd


L=[]
X=[]
Y=[]

with open('C1polar00000.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        L.append(row)

for i in range(len(L)):
    x = L[i]
    for a in x[0].split(","):
        if a != "":
            if len(a) < 13:
                Y.append(a)
            else:
                X.append(a)


"""Ces deux listes sont les données brutes"""
Xfloat = [float(k) for k in X]
Yfloat = [float(k) for k in Y]

#plt.plot(Xfloat, Yfloat)
#plt.show()

"""Yfloatfiltre est 0 si la valeure réelle est <1, et 1 sinon"""
Yfloatflitre = []
for x in Yfloat:
    if x>1:
        Yfloatflitre.append(1)
    else:
        Yfloatflitre.append(0)

#plt.plot(Xfloat, Yfloatflitre)
#plt.show()

"""deltaT est la longueur est phases (haut ou bas), chaque changement est donc une transition"""
deltaT = []
compteur = 1
for i in range(1, len(Yfloatflitre)):
    if Yfloatflitre[i] == Yfloatflitre[i-1]:
        compteur += 1
    else: 
        if Yfloatflitre[i] == 0:
            deltaT.append(compteur)
        else: 
            deltaT.append(-compteur)
        compteur = 1
if Yfloatflitre[-1] == 0:
    deltaT.append(-compteur)
else: 
    deltaT.append(compteur)


'''donnees = pd.read_csv("C1polar00000.csv", header = None)
donnees = donnees.drop([2], axis=1)
donnees = donnees.drop([1], axis=1)
donnees = donnees.drop([0], axis=1)
abs = donnees[3].values
ord = donnees[4].values 
#ord = np.histogram(ord, bins=len(ord), density=True)[0]

state = []
for i in ord:
    if i<1 :
        state.append(0)
    elif i>1 :
        state.append(1)'''

transition = pd.Series(deltaT)


class ComplexityEntropy:
    '''Class containing appropriate functions to calculate the complexity and
    entropy values for a given time series'''

    def __init__(self, time_series, d=5, tau=1):
        '''
        Parameters
        ==========
        time_series : list / array
            Time series
        d : int
            Embedding dimension
        tau : int, optional
            Embedding delay. The default is 1.
            
        Returns
        =======
        None. Initializing function
        '''
        self.time_series = np.array(time_series)
        self.d = d
        self.tau = tau



    def Permutation_Frequency(self):
        """
        Function that calculates the relative frequency for the different permutations
        of the time series for the given embedding dimension.

        Returns
        =======
        relative frequency/relative frequency pad: ARRAY
        Probability distribution for the possible ordinal patterns, padded with zeros to have length d!
        """

        possible_permutations = math.factorial(self.d)
        perm_list = []

        # subsample the time series
        self.time_series = self.time_series[::self.tau]

        for i in range(len(self.time_series)-(self.d+1)):
            # permutation of dimension-sized segments of the time series
            permutation = list(np.argsort(self.time_series[i:(self.d+i)]))
            perm_list.append(permutation)

        # Find the different permutations and calculates their number of appearance
        elements, frequency = np.unique(np.array(perm_list), return_counts=True, axis=0)

        # Divides by the total number of permutations, gets relative frequency/"probability" of appearance
        relative_frequency = np.divide(frequency, (len(self.time_series) - self.tau*(self.d-1)))

        # If the two arrays do not have the same shape, add zero padding to make their lengths equal
        if len(relative_frequency) != possible_permutations:
            relative_frequency_pad = np.pad(relative_frequency, (0, int(possible_permutations-len(relative_frequency))), mode='constant')
            return relative_frequency_pad
        else:
            return relative_frequency

    def PermutationEntropy(self, PermutationProbability):

        '''
        Function to calculate the permutation entropy for a given probability distribution.
        In this case, it returns the normalized Shannon entropy.

        Parameters
        ==========
        PermutationProbability: ARRAY
            Array containing the probability distribution of the ordinal patterns.

        Returns
        =======
        permutation_entropy: FLOAT
            Entropy value of the time series.

        '''
        permutation_entropy = 0.0

        # Calculate the max entropy, max = log(d!)
        max_entropy = np.log2(len(PermutationProbability))

        for p in PermutationProbability:
            if p != 0.0:
                permutation_entropy += p * np.log2(p)

        permutation_entropy /= max_entropy

        return -permutation_entropy


    def ShannonEntropy(self, PermutationProbability):
        '''
        Regular Shannon entropy, not normalized.

        Parameters
        ==========
        PermutationProbability: ARRAY
            Array containing the probability distribution of the ordinal patterns.

        Returns
        =======
        shannon_entropy: FLOAT
            Shannon entropy value.
        '''
        shannon_entropy = 0.0

        for p in PermutationProbability:
            if p != 0.0:
                shannon_entropy += p * np.log2(p)
        #print("bb")

        return -shannon_entropy

    def JensenShannonComplexity(self, PermutationProbability):
        '''
        Function to calculate the Jensen-Shannon complexity value for the time series
        
        Parameters
        ==========
        PermutationProbability: ARRAY
            Array containing the probability distribution of the ordinal patterns.
            
        Returns
        =======
        jensenshannoncomplexity: FLOAT
            Jensen-Shannon complexity value.
        '''
        P = PermutationProbability
        N = len(P)
        
        C1 = (N + 1) / N * np.log2(N + 1)
        C2 = 2 * np.log2(2 * N)
        C3 = np.log2(N)
        
        PE = self.PermutationEntropy(P)
        
        Puniform = []
        for i in range(N):
            Puniform.append(1 / N)
        Pk = (P+ Puniform)
        JSdiv = self.ShannonEntropy(Pk *0.5 ) - 0.5 * self.ShannonEntropy(P) - 0.5 * self.ShannonEntropy(Puniform)
        
        jensenshannoncomplexity = -2 * (1 / (C1 - C2 + C3)) * JSdiv * PE
        
        
        return jensenshannoncomplexity

    def CHplane(self):
        '''
        Computes the permutation entropy and the Jensen-Shannon complexity for the time series
        with the functions defined in the class

        Returns:
        ========
        permutation_entropy: FLOAT
            Permutation entropy value of the time series.
        jensen_shannon_complexity: FLOAT
            Jensen-Shannon complexity value of the time series.
        '''
        # Calling the function to generate the relative frequency for the ordinal patterns
        relative_frequency = self.Permutation_Frequency()
        #print(relative_frequency)

        # Using relative frequency to calculate the entropy/complexity for the time series
        permutation_entropy = self.PermutationEntropy(relative_frequency)
        jensen_shannon_complexity = self.JensenShannonComplexity(relative_frequency)
        

        return permutation_entropy, jensen_shannon_complexity, relative_frequency

class MaxMinComplexity(ComplexityEntropy):
    '''
    Class containing the functions to calculate the maximum complexity and
    minimum complexity lines for the Complexity-Entropy plane with
    embedding dimension d
    '''

    def __init__(self, d, nsteps=500):
        '''
        Parameters
        ==========
        d : int
            Embedding dimension.

        Returns
        =======
        None.
        '''

        # Initializing instance variables available to all functions/methods contained in this class
        self.d = d
        #print(self.d)
        self.N = math.factorial(self.d)
        #print(self.N)
        self.nsteps = nsteps
        self.dstep = (1-1/self.N) / (self.nsteps)

        # Initializing init()=function to parent (ComplexityEntropy class) class to make the functions contained in that class available to this class
        super().__init__(time_series= None, d= self.d)

        # Lists to contain the x and y values for the minimum and maximum complexity lines
        self.min_complexity_entropy_x = list()
        self.min_complexity_entropy_y = list()
        self.max_complexity_entropy_x = list()
        self.max_complexity_entropy_y = list()
        
    def Minimum(self):
        #print("aa")
        '''
        Function to calculate the minimum complexity line

        Returns
        =======
        min_complexity_entropy_x: list
            x values for the minimum complexity line.
        min_complexity_entropy_y: list
            y values for the minimum complexity line.
        '''
        p_min = list(np.arange(1/self.N, 1, self.dstep))

        for n in tqdm(range(len(p_min)), desc='Minimum', ncols=70):
            p_minimize = []
            if p_min[n] > 1:
                p_min[n] = 1
            p_minimize.append(p_min[n])
            for i in range(self.N-1):
                p_rest = (1-p_min[n])/(self.N-1)
                p_minimize.append(p_rest)
            p_minimize = np.array(p_minimize)
            self.min_complexity_entropy_x.append(self.PermutationEntropy(p_minimize))
            self.min_complexity_entropy_y.append(self.JensenShannonComplexity(p_minimize))
        return self.min_complexity_entropy_x, self.min_complexity_entropy_y


    def Maximum(self):
        #print("333")
        '''
        Function to calculate the maximum complexity line

        Returns
        =======
        max_complexity_entropy_x: float
            x values for the maximum complexity line.
        max_complexity_entropy_y: float
            y values for the maximum complexity line.
        '''
        
  
        for n in tqdm(range(self.N -1 ), desc='Maximum', ncols=80):
            p_max = list(np.arange(0, 1/(self.N-n), self.dstep))
            
            for m in range(len(p_max)):
                p_maximize = list()
                p_maximize.append(p_max[m])
                p_rest = (1-p_max[m])/(self.N - n -1)
                for i in range(self.N - n - 1):
                    p_maximize.append(p_rest)
                if len(p_maximize) != self.N:
                    p_maximize = np.pad(p_maximize, (0, n), mode='constant')
                p_maximize = np.array(p_maximize)
                self.max_complexity_entropy_x.append(self.PermutationEntropy(p_maximize))
                self.max_complexity_entropy_y.append(self.JensenShannonComplexity(p_maximize))
        return self.max_complexity_entropy_x, self.max_complexity_entropy_y
        

        

t = np.linspace(0, np.pi, 1000)  # Génère un vecteur temps de 0 à 2pi avec 1000 échantillons
f = 10  # Pulsation de 10 rad/s
y = np.sin(f * t)  # Calcule les valeurs du sinus
time_series= np.random.rand(1000)

transitions = transition.values
a=MaxMinComplexity(5).Minimum()
b= MaxMinComplexity(5).Maximum()
c =ComplexityEntropy(time_series).CHplane()
d =ComplexityEntropy(transitions).CHplane()
e = ComplexityEntropy(y).CHplane()

plt.scatter(e[0],e[1])
plt.scatter(d[0],d[1])
plt.scatter(c[0],c[1],marker = "s")
plt.plot(b[0],b[1])
plt.plot(a[0],a[1])
plt.xlabel("Entropie de permutation")
plt.ylabel("Complexité statistique")
plt.legend(["sinus","transitions GC","bruit blanc"])
plt.show()




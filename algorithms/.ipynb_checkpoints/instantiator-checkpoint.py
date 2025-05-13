from algorithms.greedy_boosted import GreedyOptAlgorithm
from algorithms.nfvsdn import Goku
from algorithms.osfem import Osfem
from algorithms.genetic_alg import Genetic
from algorithms.random_algorithm import RandomAlgorithm
from algorithms.greedy_algorithm import GreedyAlgorithm
from algorithms.dynamic_programming_algorithm import DynamicProgrammingAlgorithm
from algorithms.k_shortest_paths_algorithm import KShortestPathsAlgorithm
from algorithms.betweenness_centrality_algorithm import BetweennessCentralityAlgorithm
from algorithms.musfico import Musfico  
from algorithms.msf import MSF
from algorithms.new_alg import NewAlg
from algorithms.bruno_alg import BrunoAlg
from algorithms.bruno_alg_2 import BrunoAlgNew
from algorithms.rodrigo_alg import Rodrigo
from algorithms.vegeta import Vegeta
from algorithms.Kuririn_Prototype import Kuririn

class AlgorithmInstantiator:
    def instantiate_algorithm(self, type):
        if type == 'musfico':
            alg = Musfico()
        elif type =='new_alg':
            alg = NewAlg()
        elif type == 'dp':
            alg = DynamicProgrammingAlgorithm()
        elif type == 'g':
            alg = GreedyAlgorithm()  
        elif type == 'greedyb':
            alg = GreedyOptAlgorithm()  
        elif type == 'k':
            alg = KShortestPathsAlgorithm(5)
        elif type == 'b':
            alg = BetweennessCentralityAlgorithm()
        elif type == 'msf':
            alg = MSF()
        elif type =='bruno':
            alg = BrunoAlg()
        elif type =='brunonew':
            alg = BrunoAlgNew()
        elif type =='rodrigo':
            alg = Rodrigo()
        elif type ==  'ga':
            alg = Genetic()
        elif type ==  'osfem':
            alg = Osfem()
        elif type ==  'goku':
            alg = Goku()
        elif type ==  'vegeta':
            alg = Vegeta()
        elif type == 'kuririn':
            alg = Kuririn('PPO')
        else:
            raise ValueError('algorithm not found')
        return alg
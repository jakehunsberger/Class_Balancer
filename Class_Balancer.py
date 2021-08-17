import numpy as np


class Class_Balancer():
    
    def __init__(self, mode='binary'):
        self.Mode = mode
        self.Convergence_Iterations = 25
        return
    
    def fit(self, Obs_IDs, Obs_Labels):
        #Obs_IDs is a numpy array of ID's
        #Obs_Labels is a one-hot encoded numpy array
        Obs_IDs, Obs_Labels = self.__Preprocess_fit_Input(Obs_IDs, Obs_Labels)
        self.__Partition_Class_Observations(Obs_IDs, Obs_Labels)
        self.__Max_Prevalence()
        self.__Tune_Betas()
        return
    
    def __Preprocess_fit_Input(self, Obs_IDs, Obs_Labels):
        #Obs_IDs is a numpy array of ID's
        #Obs_Labels is a one-hot (or probability) encoded numpy array
        
        #Check Obs_IDs data type
        if type(Obs_IDs) is not np.ndarray:
            print('\nClass Balancer: Obs_IDs argument expected numpy.ndarray but received '+str(type(Obs_IDs))+' instead\n')
            return
        
        #Check Obs_Labels data type
        if type(Obs_Labels) is not np.ndarray:
            print('\nClass Balancer: Obs_Labels argument expected numpy.ndarray but received '+str(type(Obs_Labels))+' instead\n')
            return
        
        #Check shape of Obs_IDs
        if len(Obs_IDs.shape) == 2:
            Obs_IDs = Obs_IDs.reshape(-1)
            pass
        elif len(Obs_IDs.shape) > 2:
            print('\nClass Balancer: Obs_IDs argument shape has more than two dimensions\n')
            return
        
        #Check shape of Obs_Labels
        if len(Obs_Labels.shape) > 2:
            print('\nClass Balancer: Obs_Labels argument shape has more than two dimensions\n')
            return
        
        #Convert binary to multi-class problem
        if self.Mode == 'binary':
            
            #Correct shape
            if len(Obs_Labels.shape) == 1:
                Obs_Labels = Obs_Labels.reshape(-1,1)
                pass
            elif len(Obs_Labels.shape) == 2:
                if Obs_Labels.shape[1] != 1:
                    print('\nClass Balancer: Mode is binary but received multi-class input array for Obs_Labels\n')
                    return
                pass
            
            #Include complementary condition as alternative class
            Complementary_Labels = 1 - Obs_Labels
            Obs_Labels = np.concatenate((Obs_Labels, Complementary_Labels), axis=1)
            pass
        
        #Check agreement of dimensions for Obs_IDs and Obs_Labels
        if len(Obs_IDs) != len(Obs_Labels):
            print('\nClass Balancer: len(Obs_IDs) != len(Obs_Labels)\n')
            return
        
        #Check whether each observation features a single class label
        Obs_Label_Counts = np.sum(Obs_Labels, axis=1)
        if not Obs_Label_Counts.all() == 1:
            
            #Error message
            print('\nClass Balancer: not all observations have a single class label... removing faulty observations\n')
            
            #Fix by removing faulty observations
            Retain_Bool = (Obs_Label_Counts == 1)
            Obs_IDs = Obs_IDs[Retain_Bool]
            Obs_Labels = Obs_Labels[Retain_Bool]
            pass
        
        return Obs_IDs, Obs_Labels
    
    
    def __Partition_Class_Observations(self, Obs_IDs, Obs_Labels):
        #Create class attribute dictionary... key: class index, value: numpy array of class Obs_IDs
        
        Partitions_dict = {}
        for Class_Index in range(Obs_Labels.shape[1]):
            Partitions_dict[Class_Index] = Obs_IDs[Obs_Labels[:,Class_Index]==1]
            continue
        
        self.Partitions_dict = Partitions_dict
        return
    
    def __Max_Prevalence(self):
        #Determine extent of highest occurring class
        
        Max_Prevalence = 0
        for Class_Key in self.Partitions_dict.keys():
            Class_Obs_Num = len(self.Partitions_dict[Class_Key])
            if Class_Obs_Num > Max_Prevalence:
                Max_Prevalence = Class_Obs_Num
                pass
            continue
        self.Max_Prevalence = Max_Prevalence
        return
    
    
    def sample(self, seed=None):
        
        #Set seed
        if seed is not None:
            np.random.seed(seed)
            pass
        
        Obs_IDs = None
        for Class_Key in self.Partitions_dict.keys():
            
            #Sample with replacement
            #For all classes except most prevalent... len(Sample_IDs) > len(Class_Obs_IDs)
            Class_Obs_IDs = self.Partitions_dict[Class_Key]
            Sample_Indices = np.random.randint(low=0, high=len(Class_Obs_IDs), size=self.Max_Prevalence)
            Sample_IDs = Class_Obs_IDs[Sample_Indices]
            
            #Concatenate with other class sample observations
            if Obs_IDs is None:
                Obs_IDs = np.copy(Sample_IDs)
                pass
            else:
                Obs_IDs = np.concatenate((Obs_IDs, Sample_IDs), axis=0)
                pass
            
            continue
        
        return Obs_IDs
    
    
    def __Tune_Betas(self):
        
        #Get class observation counts
        Class_Obs_Counts = []
        for Class_Key in self.Partitions_dict.keys():
            Class_Obs_Counts.append(len(self.Partitions_dict[Class_Key]))
            continue
        Class_Obs_Counts = np.array(Class_Obs_Counts)
        
        #Class probabilities
        Class_Probs = Class_Obs_Counts / np.sum(Class_Obs_Counts)
        
        #Initialize betas
        Betas = np.zeros(shape=len(Class_Probs))
        
        #Loop predefined amount for convergence
        for i in range(self.Convergence_Iterations):
            
            #Softmax partition function
            Q = np.sum(np.e**Betas)
            
            #Algebraic solution for each beta value
            Betas = np.log((Class_Probs/(1 - Class_Probs)) * (Q - np.e**Betas))
            
            #Arbitrarily set last beta to zero to avoid diverging solution
            Betas[len(Betas)-1] = 0
            continue
        
        #Set class attribute betas
        self.Betas = np.copy(Betas)
        return
    
    
    def recover(self, Y_Probs):
        
        #Determine whether Y_Probs is appropriate input
        Y_Probs = self.__Preprocess_recover_Input(Y_Probs)
        
        #Compute log odds that yield the balanced probability estimates
        Log_Odds = self.__Log_Odds(Y_Probs)
        
        #Utilized tuned betas to recover imbalanced probability estimates
        Recovered_Probs = self.__Recovered_Probs(Log_Odds)
        
        #Cater to binary condition
        if self.Mode == 'binary':
            Recovered_Probs = Recovered_Probs[:,0]
            pass
        
        return Recovered_Probs
    
    
    def __Preprocess_recover_Input(self, Y_Probs):
        
        #Check Y_Probs data type
        if type(Y_Probs) is not np.ndarray:
            print('\nClass Balancer, recover: Y_Probs argument expected numpy.ndarray but received '+str(type(Y_Probs))+' instead\n')
            return
        
        #Convert binary to multi-class problem
        if self.Mode == 'binary':
            
            #Check shape
            if len(Y_Probs.shape) == 1:
                Y_Probs = Y_Probs.reshape(-1,1)
                pass
            elif len(Y_Probs.shape) == 2:
                if Y_Probs.shape[1] > 1:
                    print('\nClass Balancer, recover: mode set to binary but multi-class input received for Y_Probs argument\n')
                    return
                pass
            else:
                print('\n\nClass Balancer, recover: invalid shape for Y_Probs\n')
                return
            
            #Create dummy column to mimick multi-class alternatives
            Complementary_Prob = 1 - Y_Probs
            Y_Probs = np.concatenate((Y_Probs, Complementary_Prob), axis=1)
            pass
        
        #Check that input classes matches training classes
        if Y_Probs.shape[1] != len(self.Betas):
            print('\nClass Balancer, recover: Y_Probs number of classes does not match that of training data Obs_Labels\n')
            return
        
        return Y_Probs
    
    
    def __Log_Odds(self, Y_Probs):
        
        #Converge
        Log_Odds = np.zeros(shape=Y_Probs.shape)
        for i in range(self.Convergence_Iterations):
            
            #Softmax partition function
            Q = np.sum(np.e**Log_Odds, axis=1).reshape(-1,1)
            
            #Algebraic solution for each log odds value
            Log_Odds = np.log((Y_Probs/(1 - Y_Probs)) * (Q - np.e**Log_Odds))
            
            #Arbitrarily set last log odd to zero to avoid diverging solution
            Log_Odds[:, Log_Odds.shape[1]-1] = 0
            continue
        
        return Log_Odds
   
    
    def __Recovered_Probs(self, Log_Odds):
        
        #Reshape for broadcast operations compatibility
        Betas = np.copy(self.Betas).reshape(1,-1)
        
        #Fundamentally recover the prior distribution for each observation
        Beta_Log_Odds = Log_Odds + Betas
        
        #Softmax partition function
        Q = np.sum(np.e**Beta_Log_Odds, axis=1).reshape(-1,1)
        
        #Compute final probabilities
        Recovered_Probs = np.e**(Beta_Log_Odds) / Q
        
        return Recovered_Probs


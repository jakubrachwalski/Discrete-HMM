class HMM_scaled:

    #initialization of the model
    def __init__(self,X,M):
        self.X = X
        self.M = M

        #number of emission states
        self.V = max(max(x) for x in self.X) + 1

        #number of samples
        self.N = len(self.X)

        #Randomlyn initialize Pi, A,B
        #Priobability of the inner states
        self.P = np.ones(self.M)/self.M
        #Probabilities within the Transition Matrix
        self.A = self._generate_markov_matrix((self.M,self.M))
        #Probability within emission matrix
        self.B = self._generate_markov_matrix((self.M,self.V))

        #calculate alpha and beta
        self.alphas = []
        self.betas = []
        #probability of the sequence x
        self.ps = np.zeros(self.N)

        #for rescaling
        self.scales = []

        #hol the cost for each instance
        self.cost = []


    #train the model.
    def train(self,loops):
        self.print()
        for l in range(loops):
            print("===================LOOP: "+str(l))
            self._calcualte_alfa_beta()
            self._BW_update_AB()
            self._BW_update_P()
            self._update_cost()
            self.print()




    #function to generate matrix of size 'size' where each rows sums to 1
    def _generate_markov_matrix(self,size):
        np.random.seed(123)
        m = np.random.random(size)
        m = m/np.sum(m,axis=1,keepdims=True)
        return m

    #FORWARD-BACKWARD algorithm

    #calcaulte alpha based on P, A, B (parameters of HMM)
    def _calculate_alpha(self,x):

        T = len(x)

        a = np.zeros((T,self.M))
        s = np.zeros(T)

        a[0] = self.P*self.B[:,x[0]]
        s[0] = a[0].sum()
        a[0] /= s[0]

        for t in range (1,T):
            a[t] = a[t-1].dot(self.A)*self.B[:,x[t]]
            s[t] = a[t].sum()
            a[t] /= s[t]

        return a,s

    #calcaulte p based on P, A, B (parameters of HMM)
    def _calcualte_logp(self,scale):
        return np.log(scale).sum()


    #calcaulte beta based on P, A, B (parameters of HMM)
    def _calculate_beta(self,x,s):
        T = len(x)

        b = np.zeros((T,self.M))

        b[-1] = 1

        for t in range (T-2,-1,-1):
            b[t] = self.A.dot(self.B[:,x[t+1]]*b[t+1])/s[t+1] #induction

        return b

    #update alphas, betas and p's
    def _calcualte_alfa_beta(self):

        #first restart parameters
        self.alphas = []
        self.betas = []
        self.ps = np.zeros(self.N)
        self.scales = []


        for n in range(self.N):
            (a,s) = self._calculate_alpha(self.X[n])
            self.alphas.append(a)
            self.scales.append(s)
            self.ps[n] = self._calcualte_logp(s)
            b = self._calculate_beta(self.X[n],s)
            self.betas.append(b)


    #BAUM-WELCH Algorithm
    #BW: calcaulte new value of P (probabilities of States)
    def _BW_update_P(self):
        self.P = np.sum((self.alphas[n][0] * self.betas[n][0]) for n in range(self.N)) / self.N


    #calcaulte  new value of AB (tranistion matrix)
    def _BW_update_AB(self):


        a_num = np.zeros((self.M,self.M))
        b_num = np.zeros((self.M,self.V))
        den1 = np.zeros((self.M, 1))
        den2 = np.zeros((self.M, 1))

        for n in range(self.N):
            x = self.X[n]
            T = len(x)

            den1 += (self.alphas[n][:-1] * self.betas[n][:-1]).sum(axis=0,keepdims=True).T
            den2 += (self.alphas[n] * self.betas[n]).sum(axis=0,keepdims=True).T

            #calcualte A
            for i in range(self.M):
                for j in range(self.M):
                    for t in range(T-1):
                        a_num[i,j] += self.alphas[n][t,i]*self.betas[n][t+1,j]*self.A[i,j]*self.B[j,x[t+1]]/self.scales[n][t+1]


            #calcaulte B
            for i in range(self.M):
                for t in range(T):
                    b_num[i,x[t]] += self.alphas[n][t,i] * self.betas[n][t,i]


        self.A = a_num/den1
        self.B = b_num/den2

    #calcaulte cost (Probability of the actual observation)
    def _update_cost(self):
        c = np.sum(self.ps)
        self.cost.append(c)



    #Print parameters
    def print(self):
        print("alfa's: "+str(np.sum(sum(self.alphas))))
        print("beta's: "+str(np.sum(sum(self.betas))))
        print("P: "+str(self.P))
        print("A: "+str(self.A))
        print("B: "+str(self.B))
        if(len(self.cost)>0):
            print("cost: "+str(self.cost[-1]))
        else:
            print("cost: "+str(self.cost))


    #calcaulte Likelyhood of single instance
    def log_likelihood(self,x):
        a,_ = self._calculate_alpha(x)
        return self._calcualte_logp(a)

    #calculate Likelyhood of multiple instances
    def log_likelihood_multiple(self,X):
        return np.array([self.log_likelihood(x) for x in X])

    #Viterbi Algorithm
    #the most probable sequence hidden states
    def probable_states(self,x):
        T = len(x)
        delta = np.zeros((T,self.M))
        psi = np.zeros ((T,self.M))
        delta[0] = np.log(self.P)+np.log(self.B[:,x[0]])

        for t in range(1,T):
            for j in range (self.M):
                delta[t,j] = np.max(delta[t-1]+np.log(self.A[:,j]))+np.log(self.B[j,x[t]])
                psi[t,j] = np.argmax(delta[t-1]+np.log(self.A[:,j]))

        states = np.zeros(T,dtype=np.int32)

        states[T-1] = np.argmax(delta[T-1])

        for t in range (T-2, -1, -1):
            states[t] = psi[t+1,states[t+1]]

        return states

    #Generate new sequence based on the model
    def generate_sequence(self,T):
        s = np.random.choice(range(self.M),p = self.P) #initial state
        x = np.random.choice(range(self.V),p = self.B[s]) #initial observation
        sequence = [x]
        for t in range(T-1):
            s = np.random.choice(range(self.M),p = self.A[s])
            x = np.random.choice(range(self.V),p = self.B[s])
            sequence.append(x)
        return sequence

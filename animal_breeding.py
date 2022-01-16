import numpy as np
import pandas as pd 
from scipy.stats import binom

class farm_problem():
    def __init__(
        self, 
        max_animals: int,
        animal_price: float, 
        feeding_cost: float,
        discount: float,
        p: float
    ):
        """
        Formulating the farm problem for an different amount of animals.

        Parameters
        ----------
        max_animals
            An integer amount of max animals to be considered.
        animal_price
            The fixed selling price of one animal.
        feeding_cost
            The marginal feeding cost.
        discount
            The discount required to sell animals in bulks.
        p
            The probability of successfully producing one offspring.

        Returns
        class
            General MDP class to handle the farm problem.
        """

        self.max_animals = max_animals
        self.animal_price = animal_price
        self.feeding_cost = feeding_cost 
        self.discount = discount
        self.discount_foo = lambda y: (1-self.discount)**(y-1)
        self.p = p

        self.states = range(0, max_animals+1)
        self.actions = dict()
        for state in self.states[1:]:
            if state > 1:
                self.actions[state] = ["W", "B", *range(1, state+1)]
            else:
                self.actions[state] = ["W", 1]
        self.actions[0] = ["W"]

        return 

    def r(self, x, a) -> float:
        """
        Reward choosing action a in state x. 

        Parameters
        ----------
        x
            state 
        a
            action

        Returns
        -------
        float
            reward from choosing action a in state x.
        """

        # If action 'Wait' is choosen in state x, you pay the feeding costs.

        # If action 'Breed' is choosen in state x, number of animals x // 2 try to breed.
        # Animals are arranged into pairs, and if they are successfull, they produce one offspring.
        # You pay the feeding costs for the current amount of animals and the success of the breeding
        # is depicted in the transition probabilities.

        if a == "W" or a == "B":
            return -1 * x * self.feeding_cost
        
        # If the action 'Sell y' for y <= x is choosen, number of animals y are sold.
        # The market can only take so many animals and selling in bulk must be done using a discount.
        # The total reward for selling the animals is (1-discount**(y-1)) * price. You do not pay 
        # for the feeding of the sold animals.

        else:
            return -1 * (x-a) * self.feeding_cost + self.animal_price * self.discount_foo(a) * a

    def R(self, x) -> float:
        """
        Exit reward function from being in state x. (Only to be considered if finite horizon problem.)

        Parameters
        ----------
        x
            state

        Returns
        -------
        float
            reward from selling off x animals.
        """

        return  self.animal_price * self.discount_foo(x) * x

    def t(self, y, x, a) -> float:
    
    
        # If action 'Wait' is choosen, we return to the same state as we came from.
        if a == "W":
            if y == x:
                return 1
            else:
                return 0

        # If action 'Breed' is choosen, we with probability p for each pair x // 2 receive one 
        # offspring. 
        elif a == "B":
            if y >= x:
                to_produce = y - x
                pairs = x // 2
                if y == self.max_animals:
                    tmp = 1 - binom.cdf(to_produce, pairs, self.p)
                    return tmp + binom.pmf(to_produce, pairs, self.p)
                else:
                    return binom.pmf(to_produce, pairs, self.p)
            else:
                return 0


        # If action 'Sell a' for a <= x is choosen, we go to state x-a.
        else:
            if x - a == y:
                return 1
            else:
                return 0
    
    def bia(self, time_horizon: int = 20) -> list:
        """
        Backward induction algorithm.

        Parameters
        ----------
        time_horizon
            Time horizon of the MDP problem.

        Returns
        -------
        list
            List with optimal policy and value function.
        """

        # First off, we study the MDP in a finite time horizon.
        # To this end, we may invoke the backward induction algorithm.
        T = time_horizon - 1
        # Create matrix to store Bellman equations evalulated at different states 
        # at different time points. Rows are time points, columns are states. 
        u = np.zeros(shape = (time_horizon, len(self.states)), dtype = float)
        pi = np.empty(shape = (time_horizon - 1, len(self.states)), dtype = object)

        for t in [*range(time_horizon)][::-1]:
            if t == T:
                for x in self.states:
                    u[t, x] = self.R(x)
            else:
                for x in self.states:
                    tmp = dict()
                    for a in self.actions[x]:
                        tmp[a] = self.r(x, a)
                        for y in self.states:
                            tmp[a] += self.t(y, x, a) * u[t+1, y]  
                    
                    max_key = max(tmp, key=tmp.get)
                    max_val = max(tmp.values())

                    u[t, x] = max_val
                    pi[t, x] = max_key
        
        return [pi, u]

    def via(
            self,
            lmd: float = .9,
            epsilon: float = 0.01,
            v_0: np.ndarray = None
        ) -> list:
        """
        Value iteration algorithm.

        Parameters
        ----------
        lmb
            Discount factor.
        epsilon
            Threshold multiplier.
        v_0
            Initial guess of the value function.

        Returns
        -------
        list
            List with epsilon optimal policy & value function and history of value function. 
        """
        
        
        # Value iteration algorithm for infinite horizon.
        threshold = epsilon * (1 - lmd) / (2 * lmd)
        v_0 = v_0 if type(v_0) == np.ndarray else np.zeros(shape = len(self.states)) 
        n = 0
        v_next = v_0
        stop_condition = True
        epsilon_optimal_policy = np.empty(shape=len(self.states), dtype = object)
        
        #History of the value function approximation.
        data = list()

        while stop_condition:
            
            v_current = v_next 
            v_next = np.empty(shape = len(self.states), dtype=float)
            n += 1
            for x in self.states:
                tmp = dict()
                for a in self.actions[x]:
                    tmp[a] = self.r(x, a)
                    for y in self.states:
                        tmp[a] += lmd * self.t(y, x, a) * v_current[y]
                max_key = max(tmp, key=tmp.get)
                max_val = max(tmp.values())
                v_next[x] = max_val
                epsilon_optimal_policy[x] = max_key
            
            data.append(v_next)
            stop_condition = np.linalg.norm(v_current - v_next, ord=1) > threshold

        history = pd.DataFrame(
            data=data, columns=self.states
        )
        history.index += 1

        print("Converged after {n} iterations".format(n=n))
        return [epsilon_optimal_policy, v_next, history]

    def y(self, x, a):
        """
        Simulate the transition probabilities ~ p( |x, a)

        Parameters
        ----------
        x
            State visited.
        a
            Action taken in the state visited.

        Returns
        -------
        state
            One realization of random variable with distribution ~ p( |x, a)
        """
        probability = [0 for _ in range(len(self.states))]
        for state in self.states:
            probability[state] = self.t(state, x, a)
        return np.random.choice(self.states, p=probability)

    def q(
        self,
        lmd: float = .5,
        threshold_n: int = 100,
        alpha = lambda k: 1/k
    ) -> list():
        """
        Q-learning algorithm. Note that probabilities are generated using function y.

        Parameters
        ----------
        lmd
            Discount factor.
        threshold_n
            Number of iterations in learning the Q-function.
        alpha
            Function used to generate the sequence determining impact of historical evaluations.
        
        Returns:
        list
            The history of the optimal value function using Q-learning.
        """

        n = 1

        value_function = np.empty(shape=len(self.states), dtype=float)
        policy = np.empty(shape=len(self.states), dtype=object)
        
        data = np.empty(shape=(threshold_n, len(self.states)), dtype=float)
        history = pd.DataFrame(
            data = data, columns = self.states, index=range(1,threshold_n+1)
        )
        del data

        q_current = dict()
        for state in self.states:
            q_current[state] = np.array([0 for _ in self.actions[state]])

        while n < threshold_n:
            
            q_next = dict()
            for state in self.states:
                tmp1 = (1-alpha(n)) * q_current[state]
                tmp2 = np.zeros(shape = tmp1.shape, dtype = float)
                i = 0 
                for action in self.actions[state]:
                    tmp2[i] = self.r(state, action) + lmd * np.max(q_current[self.y(state, action)]) 
                    i += 1
                tmp2 *= alpha(n)
                tmp = tmp1 + tmp2
                q_next[state] = tmp

            q_current = q_next
            for state in self.states:
                policy[state] = self.actions[state][np.argmax(q_current[state])]
                value_function[state] = max(q_current[state])
            history.loc[n,] = value_function
            n += 1

        return [policy, value_function, history]

if __name__ == "__main__":
    
    max_animals = 15 
    animal_price = 10
    feeding_cost = 1
    discount = 10 / 100
    discount_foo = lambda y: (1-discount)**(y-1)
    p = .8

    mdp = farm_problem(
        max_animals,
        animal_price,
        feeding_cost,
        discount,
        p
    )




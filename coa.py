import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from typing import Callable, Tuple

# -----------------------------
# Utility Helpers (Required by COA)
# -----------------------------

def lhs_sampling(n_samples: int, dim: int, low: float, high: float, rng: np.random.RandomState):
    """Generates n_samples using Latin Hypercube Sampling."""
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.rand(n_samples, dim)
    a, b = cut[:n_samples], cut[1:n_samples + 1]
    rdpoints = a[:, None] + (b - a)[:, None] * u
    samples = np.zeros_like(rdpoints)
    for j in range(dim):
        order = rng.permutation(n_samples)
        samples[:, j] = rdpoints[order, 0]
    return low + samples * (high - low)

def pairwise_distances(pop: np.ndarray):
    """Computes pairwise Euclidean distances between all solutions in the population."""
    diffs = pop[:, None, :] - pop[None, :, :]
    return np.linalg.norm(diffs, axis=2)

# -----------------------------
# Abstract Problem Definitions (Required by COA)
# -----------------------------

class OptimizationProblem(ABC):
    """Abstract base class for defining an optimization problem."""
    @abstractmethod
    def calculate_fitness(self, solution): pass
    
    @property
    @abstractmethod
    def dimension(self): pass

    @property
    @abstractmethod
    def bounds(self): pass

class DynamicOptimizationProblem(OptimizationProblem):
    """
    An abstract problem where fitness depends on time 't'.
    This is required for the Environmental Adaptation feature[cite: 88, 164].
    """
    @abstractmethod
    def calculate_fitness(self, solution, t: int):
        """Calculates fitness, which may change based on time/generation 't'."""
        pass

# ----------------------------------------------------
# Camouflage Optimization Algorithm (COA)
# ----------------------------------------------------

class CamouflageOptimizer:
    """
    An implementation of the Camouflage Optimization Algorithm (COA)
    that includes all 5 novelties from the documentation:
    1. Camouflage Density Function 
    2. Camouflage Escape Mechanism 
    3. Dynamic Blending Strategy 
    4. Predator-Prey Co-evolution 
    5. Environmental Adaptation Mechanism 
    """
    def __init__(self,
                 problem: DynamicOptimizationProblem, # Must be a dynamic problem
                 pop_size: int = 100,
                 max_nfe: int = None,
                 initial_predator_pop_size: int = 20,
                 min_predators: int = 5,
                 max_predators: int = 35,
                 initial_lambda: float = 0.9,
                 decay_rate: float = 2.5,
                 stagnation_limit: int = 200,
                 alpha_escape: float = 0.6,
                 beta_escape: float = 0.4,
                 n_jobs: int = -1,
                 seed: int = None,
                 elite_size: int = 5,
                 p_best_rate: float = 0.1,
                 verbose: bool = False):
        
        if not isinstance(problem, DynamicOptimizationProblem):
            raise TypeError("This COA implementation requires a DynamicOptimizationProblem to use all features.")
        
        self.problem = problem
        self.dim = problem.dimension
        self.pop_size = pop_size
        self.max_nfe = max_nfe if max_nfe else 10000 * self.dim
        self.generations = int(self.max_nfe / self.pop_size)
        
        # Predator-Prey Co-evolution parameters 
        self.predator_pop_size = initial_predator_pop_size
        self.min_predators = min_predators
        self.max_predators = max_predators

        # Blending and Escape parameters
        self.initial_lambda = initial_lambda # [cite: 142]
        self.decay_rate = decay_rate         # [cite: 143]
        self.stagnation_limit = stagnation_limit
        self.alpha_escape = alpha_escape     # [cite: 130]
        self.beta_escape = beta_escape       # [cite: 130]
        
        # General and performance parameters
        self.n_jobs = n_jobs
        self.seed = seed
        self.elite_size = max(1, elite_size)
        self.p_best_rate = p_best_rate
        self.verbose = verbose
        
        # Internal state
        self.rng = np.random.RandomState(seed)
        self.low, self.high = problem.bounds
        self.range = self.high - self.low
        self.nfe_count = 0

    def _update_nfe(self, count=1):
        """Atomically updates the number of function evaluations."""
        self.nfe_count += count

    def _camouflage_density(self, distances, fit, k=5):
        """Calculates the Camouflage Density Function θ(x) [cite: 109-115]."""
        n = distances.shape[0]
        theta = np.zeros(n)
        
        fit_range = np.max(fit) - np.min(fit)
        norm_fit = (fit - np.min(fit)) / (fit_range + 1e-12)

        for i in range(n):
            dists = np.delete(distances[i], i)
            nearest_indices = np.argpartition(dists, k)[:k]
            d_mean = np.mean(dists[nearest_indices]) # d(x) [cite: 117]
            # Use local fitness diff as proxy for gradient g(x) [cite: 118]
            g = np.abs(norm_fit[i] - np.mean(norm_fit[np.argsort(dists)[:k]]))
            
            norm_d = d_mean / self.range
            val = norm_d - g
            exp_arg = -10.0 * val # 'k' sensitivity param [cite: 119]
            theta[i] = 1.0 / (1.0 + np.exp(np.clip(exp_arg, -700, 700))) # [cite: 115]
        return theta

    def _risk(self, fitness_vals, distances, theta):
        """Calculates the Camouflage Escape Mechanism risk R(x) [cite: 126-127]."""
        risk_vals = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            d_i = np.copy(distances[i])
            d_i[i] = np.inf
            idx_neighbor = np.argmin(d_i)
            
            fit_diff = fitness_vals[i] - fitness_vals[idx_neighbor] # f(x) - f(x')
            dist_to_neighbor = distances[i, idx_neighbor]           # |x - x'|
            grad = fit_diff / (dist_to_neighbor + 1e-12)
            stagnation = 1.0 - np.tanh(grad) # Proxy for (1 - grad) [cite: 127]
            
            risk_i = self.alpha_escape * stagnation + self.beta_escape * theta[i] # [cite: 127]
            risk_vals[i] = risk_i
            
        return np.clip(risk_vals / (self.alpha_escape + self.beta_escape), 0, 1)

    def _adaptive_local_search(self, solution, t: int):
        """A local search for predators (y(t)) to 'hunt' prey (x(t)) [cite: 151-152, 158]."""
        best = solution.copy()
        best_f = self.problem.calculate_fitness(best, t) # Pass 't'
        self._update_nfe()
        
        step = self.range * 0.01
        max_iter, no_improve_limit = 30, 5
        no_improve_count = 0
        
        for _ in range(max_iter):
            if self.nfe_count >= self.max_nfe: break
            
            cand = best + self.rng.uniform(-step, step, self.dim)
            np.clip(cand, self.low, self.high, out=cand)
            cf = self.problem.calculate_fitness(cand, t) # Pass 't'
            self._update_nfe()

            if cf < best_f:
                best, best_f = cand, cf
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= no_improve_limit:
                    step *= 0.5 # Adapt step size when stuck
                    no_improve_count = 0
        return best, best_f

    def solve(self):
        """Executes the Camouflage Optimization Algorithm."""
        # [cite: 20]
        prey_pop = lhs_sampling(self.pop_size, self.dim, self.low, self.high, self.rng)
        
        # Initial fitness evaluation uses t=0 [cite: 22]
        fitness_vals = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self.problem.calculate_fitness)(x, 0) for x in prey_pop
        ))
        self._update_nfe(self.pop_size)

        best_solution, best_fitness_val = None, float('inf')
        stagnation_counter = 0

        for t in range(self.generations):
            if self.nfe_count >= self.max_nfe:
                if self.verbose: print(f"NFE limit reached at gen {t}. Stopping.")
                break

            # --- ENVIRONMENTAL ADAPTATION MECHANISM [cite: 165, 172] ---
            # We detect the change by re-evaluating one solution.
            # If its fitness has changed, the environment has shifted.
            detector_fit_old = fitness_vals[0]
            detector_fit_new = self.problem.calculate_fitness(prey_pop[0], t) # Pass current 't' [cite: 167]
            self._update_nfe()

            if np.abs(detector_fit_old - detector_fit_new) > 1e-8:
                if self.verbose: 
                    print(f"Gen {t}: Change detected! Old={detector_fit_old:.2f}, New={detector_fit_new:.2f}. Re-adapting...")
                
                # --- ADAPTATION STEP ---
                # 1. Re-evaluate the entire population in the new environment [cite: 33]
                fitness_vals = np.array(Parallel(n_jobs=self.n_jobs)(
                    delayed(self.problem.calculate_fitness)(x, t) for x in prey_pop
                ))
                self._update_nfe(self.pop_size)
                
                # 2. Reset stagnation and predator/prey dynamic
                stagnation_counter = 0
                self.predator_pop_size = self.min_predators # Reset predators
            else:
                # If no change, just update the detector's fitness
                fitness_vals[0] = detector_fit_new
            # --- (End of Adaptation) ---

            sorted_idx = np.argsort(fitness_vals)
            current_best_idx = sorted_idx[0]
            
            # --- Adaptive Predator Population (Co-evolutionary Dynamic)  ---
            if fitness_vals[current_best_idx] < best_fitness_val:
                # Prey are improving, increase predators to hunt better
                self.predator_pop_size = min(self.max_predators, self.predator_pop_size + 1)
                best_fitness_val = fitness_vals[current_best_idx]
                best_solution = prey_pop[current_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter % 20 == 0:
                    # Prey are stagnating, decrease predators
                    self.predator_pop_size = max(self.min_predators, self.predator_pop_size - 1)

            if stagnation_counter > self.stagnation_limit:
                if self.verbose: print(f"Early stop at gen {t} (stagnation).")
                break # [cite: 94]

            distances = pairwise_distances(prey_pop)
            theta = self._camouflage_density(distances, fitness_vals) # Novelty 1 [cite: 175]
            risk = self._risk(fitness_vals, distances, theta) # Novelty 2 [cite: 176]
            
            # Dynamic Blending Strategy (Novelty 3) [cite: 177]
            blend = self.initial_lambda * np.exp(-self.decay_rate * (self.nfe_count / self.max_nfe)) # [cite: 140]
            
            # --- Predator Hunting Phase (Novelty 4) [cite: 178] ---
            num_targets = min(self.predator_pop_size, len(sorted_idx))
            predator_targets_idx = self.rng.choice(
                sorted_idx[:max(num_targets, self.pop_size // 5)], 
                num_targets, 
                replace=False
            )
            # Predators (local search) hunt the prey (solutions)
            hunted_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._adaptive_local_search)(prey_pop[idx], t) for idx in predator_targets_idx # Pass 't'
            )
            for i, (hunted_sol, hunted_fit) in enumerate(hunted_results):
                target_idx = predator_targets_idx[i]
                if hunted_fit < fitness_vals[target_idx]:
                    prey_pop[target_idx], fitness_vals[target_idx] = hunted_sol, hunted_fit
            
            # --- Prey Evolution Phase ---
            new_prey_pop = np.zeros_like(prey_pop)
            # Elitism (Survival of the Fittest) [cite: 22, 61]
            elites_idx = sorted_idx[:self.elite_size]
            new_prey_pop[:self.elite_size] = prey_pop[elites_idx].copy()
            num_p_best = int(self.pop_size * self.p_best_rate)
            p_best_indices = sorted_idx[:max(1, num_p_best)]

            for i in range(self.elite_size, self.pop_size):
                sol = prey_pop[i].copy()
                
                # Escape Mechanism (Disruptive Pattern) [cite: 125, 131]
                if self.rng.rand() < risk[i]: 
                    jump_strength = 0.2 * self.range
                    # Jump to a new random location, or near the best [cite: 133-134]
                    sol = best_solution + self.rng.uniform(-jump_strength, jump_strength, self.dim) if self.rng.rand() < 0.5 else self.rng.uniform(self.low, self.high, self.dim)
                else: 
                    # Blending and Mimicry [cite: 68-69, 75]
                    pbest_idx = self.rng.choice(p_best_indices)
                    pbest_sol = prey_pop[pbest_idx]
                    r1, r2 = self.rng.choice(self.pop_size, 2, replace=False)
                    # Mimic vector [cite: 77] blended with differential vector
                    mimic_vec = (pbest_sol - sol) + (prey_pop[r1] - prey_pop[r2])
                    sol += blend * self.rng.rand() * mimic_vec # [cite: 137-138]

                # Mutation (Disruptive Patterns) [cite: 31, 79-81]
                if self.rng.rand() < (1.0 - theta[i]): # Less dense (worse) solutions mutate more
                    mutation_strength = 0.01 * self.range
                    sol += self.rng.normal(0, mutation_strength, self.dim) # [cite: 83-84]

                np.clip(sol, self.low, self.high, out=sol)
                new_prey_pop[i] = sol
            
            prey_pop = new_prey_pop
            
            # --- Final fitness evaluation of new population [cite: 24-27] ---
            fitness_vals[self.elite_size:] = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(self.problem.calculate_fitness)(x, t) for x in prey_pop[self.elite_size:] # Pass 't'
            ))
            self._update_nfe(self.pop_size - self.elite_size)

            if self.verbose and t % 50 == 0:
                print(f"Gen {t:4d} | NFE: {self.nfe_count:6d} | Preds: {self.predator_pop_size:2d} | Best: {best_fitness_val:.6e}")
        
        # [cite: 196]
        return best_solution, best_fitness_val
Camouflage Optimization Algorithm (COA)

The Camouflage Optimization Algorithm (COA) is a population based metaheuristic inspired by camouflage and survival behavior in nature. It is designed for continuous optimization problems and explicitly supports dynamic environments where the fitness landscape may change over time.

This repository contains a full Python implementation of COA with predator–prey co evolution, adaptive escape behavior, and environmental adaptation.

Key Features
1. Population based continuous optimization
2. Support for dynamic (time dependent) objective functions
3. Predator–prey co evolution mechanism
4. Camouflage density driven exploration
5. Escape mechanism to avoid stagnation
6. Dynamic blending strategy for balanced exploration and exploitation
7. Parallel fitness evaluation using joblib
8. Elitism and adaptive predator population control

Install required packages using:
pip install -r requirements.txt

Installation
Clone the repository:
git clone https://github.com/your-username/camouflage-optimization-algorithm.git
cd camouflage-optimization-algorithm

Algorithm Overview

COA maintains a population of prey solutions that evolve over generations. Their survival depends on camouflage density, local fitness structure, and predator pressure. Predators perform adaptive local searches on promising prey, while prey evolve using blending, mutation, and escape behaviors.

The algorithm includes:
1. Camouflage Density Function
2. Camouflage Escape Mechanism
3. Dynamic Blending Strategy
4. Predator–Prey Co evolution
5. Environmental Adaptation Mechanism

Important Parameters:
| Parameter                   | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `pop_size`                  | Number of prey solutions                          |
| `max_nfe`                   | Maximum number of fitness evaluations             |
| `initial_predator_pop_size` | Initial number of predators                       |
| `min_predators`             | Minimum predator population                       |
| `max_predators`             | Maximum predator population                       |
| `elite_size`                | Number of elite solutions preserved               |
| `p_best_rate`               | Fraction of best solutions used for guidance      |
| `alpha_escape`              | Weight for stagnation based escape                |
| `beta_escape`               | Weight for camouflage based escape                |
| `n_jobs`                    | Number of parallel jobs (`-1` uses all CPU cores) |
| `seed`                      | Random seed for reproducibility                   |

Output:
The solve() method returns:
1. best_solution – Best solution vector found
2. best_fitness – Fitness value of the best solution
The algorithm assumes minimization. For maximization problems, negate the objective function.

Notes and Limitations
1. This implementation requires a dynamic problem interface, even for static problems. If your problem is static, simply ignore the time parameter t.
2. All dimensions share the same lower and upper bounds.
3. Performance depends on population size, predator settings, and problem complexity.

Example Use Cases
1. Benchmark function optimization
2. Dynamic optimization problems
3. Engineering design optimization
4. Machine learning hyperparameter tuning

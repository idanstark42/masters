import numpy as np
import matplotlib.pyplot as plt

class Ising1D:
    def __init__(self, J, beta, N):
        self.J = J
        self.beta = beta
        self.T = 1 / beta
        self.N = N

    def transfer_matrix(self):
        return np.array(
            [
                [np.exp(self.beta * self.J), np.exp(-self.beta * self.J)],
                [np.exp(-self.beta * self.J), np.exp(self.beta * self.J)],
            ]
        )
    
    def eigenvalues(self):
        eigenvalues, _ = np.linalg.eig(self.transfer_matrix())
        eigenvalues = np.sort(eigenvalues)[::-1] 
        return eigenvalues
    
    def energy(self, state):
        spins = [(1 if (state >> i) & 1 else -1) for i in range(self.N)]
        return sum([- self.J * spins[i] * spins[(i + 1) % self.N] for i in range(self.N)])

    # calculate the partition function directly from the sum over all states
    def partition_function(self):
        return sum([np.exp(-self.beta * self.energy(state)) for state in range(2**self.N)])

    def free_energy_stat_mechanics(self):
        return -self.T * np.log(self.partition_function())
    
    def free_energy_approx(self):
        return -self.T * self.N * np.log(self.eigenvalues()[0])

def run(J, beta_min, beta_max, beta_steps, N_min, N_max):
    betas = []
    Ts = []
    Ns = []
    F_exacts = []
    F_approxs = []
    diffs = []

    print("beta   N    F_exact     F_approx    diff")
    print("=============================================")
    for beta in range(int(beta_min * 1000), int(beta_max * 1000) + 1, int((beta_max - beta_min) * 1000 / beta_steps)):
        beta = beta / 1000.0
        for N in range(N_min, N_max + 1):
            ising_model = Ising1D(J, beta, N)
            F_exact = ising_model.free_energy_stat_mechanics()
            F_approx = ising_model.free_energy_approx()
            print(f"{beta}  {N}  {F_exact}   {F_approx}  {abs(F_exact - F_approx)}")
            betas.append(beta)
            Ts.append(1 / beta)
            Ns.append(N)
            F_exacts.append(F_exact)
            F_approxs.append(F_approx)
            diffs.append(abs(F_exact - F_approx))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(Ts, diffs, c=Ns, cmap='viridis', alpha=0.7)

    plt.xlabel('Temperature (kT)')
    plt.ylabel('Difference |F_exact - F_approx|')
    plt.title('Difference between Exact and Approximate Free Energy vs Beta - ' + ('Ferromagnetic' if J > 0 else 'Antiferromagnetic'))
    plt.grid()
    cbar = plt.colorbar(scatter)
    cbar.set_label('System Size N')
    # save and show the figure
    plt.savefig(f'ising_1d_diff_T_{'ferro' if J > 0 else 'antiferro'}_{beta_min}_{beta_max}.png')
    plt.show()

if __name__ == "__main__":
    J = 1  
    beta_min = 0.1
    beta_max1 = 1.0
    beta_max2 = 10.0
    beta_steps = 100
    N_min = 6
    N_max = 12

    run(J, beta_min, beta_max1, beta_steps, N_min, N_max)
    run(-J, beta_min, beta_max1, beta_steps, N_min, N_max)
    run(J, beta_min, beta_max2, beta_steps, N_min, N_max)
    run(-J, beta_min, beta_max2, beta_steps, N_min, N_max)
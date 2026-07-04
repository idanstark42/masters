import numpy as np
import matplotlib.pyplot as plt
import tqdm

def random_matrix_c (n, distribution=None):
    """Returns a random n x n matrix over the complex numbers."""
    import random
    if distribution is None:
        distribution = random.random
    return [[complex(distribution(), distribution()) for _ in range(n)] for _ in range(n)]

def generate_single_distribution():
    mean = 0
    variance = 0.5
    n = 64
    population_size = 10000

    distribution = lambda: np.random.normal(mean, np.sqrt(variance))
    eigenvalues = np.array([])  # Initialize an empty array to store eigenvalues
    for _ in tqdm.tqdm(range(population_size)):
        matrix = random_matrix_c(n, distribution)
        eigenvalues = np.concatenate((eigenvalues, np.linalg.eigvals(matrix)))  # Append eigenvalues to the array

    # plot and save the distribution of real and imaginary parts of the eigenvalues
    plt.figure(figsize=(8, 8))
    plt.hist(eigenvalues.real, bins=100, alpha=0.5, label='Real Part', density=True)
    plt.hist(eigenvalues.imag, bins=100, alpha=0.5, label='Imaginary Part', density=True)
    plt.title('Distribution of Eigenvalues of Random Complex Matrices')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.savefig('figures/eigenvalue_distribution.png')
    plt.show()
    # plot and save the distribution of eigenvalue magnitudes
    plt.figure(figsize=(8, 8))
    plt.hist(np.abs(eigenvalues), bins=100, alpha=0.5, label='Magnitude', density=True)
    plt.title('Distribution of Eigenvalue Magnitudes of Random Complex Matrices')
    plt.xlabel('Magnitude')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.savefig('figures/eigenvalue_magnitude_distribution.png')
    plt.show()
    # plot and save the distribution of eigenvalue phases
    plt.figure(figsize=(8, 8))
    plt.hist(np.angle(eigenvalues), bins=100, alpha=0.5, label='Phase', density=True)
    plt.title('Distribution of Eigenvalue Phases of Random Complex Matrices')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.savefig('figures/eigenvalue_phase_distribution.png')
    plt.show()
    # plot and save the distribution of eigenvalues in the complex plane
    plt.figure(figsize=(8, 8))
    plt.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.5)
    plt.title('Distribution of Eigenvalues of Random Complex Matrices in the Complex Plane')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid()
    plt.savefig('figures/eigenvalue_complex_plane_distribution.png')
    plt.show()

def generate_radius_vs_n ():
    mean = 0
    variance = 0.5
    population_size = 2
    n_values = range(2, 500, 10) # range of n values to test: from 10 to 1000 with step of 10
    radii = []
    distribution = lambda: np.random.normal(mean, np.sqrt(variance))
    tquote = tqdm.tqdm(total=len(n_values) * population_size)
    for n in n_values:
        eigenvalues = np.array([])  # Initialize an empty array to store eigenvalues
        for _ in range(population_size):
            matrix = random_matrix_c(n, distribution)
            eigenvalues = np.concatenate((eigenvalues, np.linalg.eigvals(matrix)))  # Append eigenvalues to the array
            tquote.update(1)
        radii.append(np.max(np.abs(eigenvalues)))  # Store the maximum radius for this n

    # plot and save the radius vs n
    plt.figure(figsize=(8, 8))
    plt.plot(n_values, radii, marker='o', label='Maximum Magnitude of Eigenvalues')
    # plot \sqrt{n} for comparison
    plt.plot(n_values, np.sqrt(n_values), label='$\\sqrt{n}$')
    plt.legend()
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Maximum Radius of Eigenvalues')
    plt.grid()
    plt.title('Maximum Radius of Eigenvalues vs Matrix Size')
    plt.savefig('figures/radius_vs_n.png')
    plt.show()

    # plot and save the radius vs n
    plt.figure(figsize=(8, 8))
    plt.plot(n_values, radii, marker='o', label='Maximum Magnitude of Eigenvalues')
    # plot \sqrt{n} for comparison
    plt.plot(n_values, np.sqrt(n_values), label='$\\sqrt{n}$')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrix Size (n) (Log Scale)')
    plt.ylabel('Maximum Radius of Eigenvalues (Log Scale)')
    plt.grid()
    plt.title('Maximum Radius of Eigenvalues vs Matrix Size (Log-Log Scale)')
    plt.savefig('figures/radius_vs_n_loglog.png')
    plt.show()

def generate_level_repulsion ():
    mean = 0
    variance = 0.5
    n = 2
    population_size = 1000000

    distribution = lambda: np.random.normal(mean, np.sqrt(variance))

    spacings = np.array([])  # Initialize an empty array to store level spacings
    for _ in tqdm.tqdm(range(population_size)):
        matrix = random_matrix_c(n, distribution)
        eigenvalues = np.linalg.eigvals(matrix)
        # compute the level spacing distribution in the matrix
        spacing = eigenvalues[1] - eigenvalues[0]  # Compute the spacing between the two eigenvalues
        spacings = np.concatenate((spacings, [np.abs(spacing)]))  # Append the absolute value of the spacing to the array

    spacings = spacings / np.mean(spacings)  # Normalize spacings by the mean spacing to make the mean spacing equal to 1

    # plot and save the level spacing distribution
    plt.figure(figsize=(8, 8))
    plt.hist(spacings, bins=100, alpha=0.5, label='Level Spacing', density=True)
    s = np.linspace(0, np.max(spacings), 100)
    plt.plot(s, 81 * np.pi ** 2 / 128 * s ** 3 * np.exp(- 9 * np.pi / 16 * s**2), label='Wigner Surmise for Ginibre Ensemble $\\beta=3$')
    plt.title('Distribution of Level Spacings of Random Complex Matrices')
    plt.xlabel('Level Spacing')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.savefig('figures/level_spacings_distribution.png')
    plt.show()

if __name__ == "__main__":
    # generate_single_distribution()
    # generate_radius_vs_n()
    generate_level_repulsion()
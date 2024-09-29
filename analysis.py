import subprocess
import os

# Define the path to the analysis script
analysis_script_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/analysis.py"
csv_files = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data/processed"
processed_data_path = "/mnt/c/Users/Shrinjita Paul/Documents/GitHub/PublicTransportationEfficiencyData/data"
# Iterate over each processed CSV file and call analysis.py
for csv_file in csv_files:
    processed_file_path = os.path.join(processed_data_path, f"cleaned_{csv_file}")
    
    # Debug: Display the file being analyzed
    print(f"Running analysis on: {processed_file_path}")
    
    # Call the analysis script with the processed file as an argument
    try:
        result = subprocess.run(
            ['python', analysis_script_path, processed_file_path],
            check=True, capture_output=True, text=True
        )
        print(f"Analysis completed for {processed_file_path}")
        print(result.stdout)  # Debug: Display output from the analysis script
    except subprocess.CalledProcessError as e:
        print(f"Error while analyzing {processed_file_path}: {e.stderr}")


def objective_function(params):
    x, y = params
    return (x - 0.5)**2 + (y - 0.5)**2

class BOA:
    def __init__(self, objective_function, num_iterations=100, population_size=50):
        self.objective_function = objective_function
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.population = np.random.rand(self.population_size, 2)

    def optimize(self):
        best_score = float('inf')
        best_params = None
        
        for _ in range(self.num_iterations):
            for i in range(self.population_size):
                score = self.objective_function(self.population[i])
                
                if score < best_score:
                    best_score = score
                    best_params = self.population[i]
                    
            self.population += np.random.normal(0, 0.1, self.population.shape)

        return best_params

X, y = generate_random_dataset()

boa_optimizer = BOA(objective_function, num_iterations=100, population_size=50)
best_params = boa_optimizer.optimize()

print("Best parameters found:", best_params)
print("Best score:", objective_function(best_params))

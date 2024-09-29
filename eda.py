import pandas as pd  # Modin for faster DataFrame operations
import seaborn as sns  # Seaborn for visualization
import matplotlib.pyplot as plt
import plotly.express as px  # Plotly for interactive visualizations

# Load dataset using Modin (change 'vehicles.csv' to your dataset path)
data = pd.read_csv('vehicles.csv')

# Perform basic data inspection
print("Basic Data Information:")
print(data.info())
print("\nFirst 5 rows of the data:")
print(data.head())

# Data cleaning - Remove duplicates and handle missing values
data_cleaned = data.drop_duplicates().fillna(data.mean())

# Create some new features for analysis
data_cleaned['emission_per_ps'] = data_cleaned['co2_emissions_gPERkm'] / data_cleaned['power_ps']
data_cleaned['engine_size_liters'] = data_cleaned['engine_size_cm3'] / 1000

# Visualization 1: Distribution of CO2 emissions using Matplotlib
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['co2_emissions_gPERkm'], bins=30, kde=True, color='blue')
plt.title('Distribution of CO2 Emissions (g/km)')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Correlation heatmap to find relationships between variables using Matplotlib
plt.figure(figsize=(12, 8))
correlation_matrix = data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Vehicle Features')
plt.show()

# Visualization 3: Engine size vs CO2 emissions with fuel type using Matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(x='engine_size_liters', y='co2_emissions_gPERkm', hue='fuel', data=data_cleaned)
plt.title('Engine Size vs CO2 Emissions by Fuel Type')
plt.xlabel('Engine Size (Liters)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend(title='Fuel Type')
plt.show()

# Visualization 4: Boxplot of CO2 emissions by transmission type using Matplotlib
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission', y='co2_emissions_gPERkm', data=data_cleaned)
plt.title('CO2 Emissions by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

# Save cleaned data to a new file for modeling purposes
data_cleaned.to_csv('vehicles_cleaned.csv', index=False)

print("Exploratory Data Analysis complete! Cleaning data saved to 'vehicles_cleaned.csv'.")

def create_visualizations():
    # Load dataset
    data = pd.read_csv('vehicles_cleaned.csv')  # Ensure correct file path

    # Create visualizations
    visualizations = []

    # Visualization 1: Distribution of CO2 emissions using Plotly
    fig1 = px.histogram(data, x='co2_emissions_gPERkm', nbins=30, title='Distribution of CO2 Emissions (g/km)', 
                         labels={'co2_emissions_gPERkm': 'CO2 Emissions (g/km)'})
    visualizations.append(fig1)

    # Visualization 2: Correlation heatmap using Plotly
    correlation_matrix = data.corr()
    fig2 = px.imshow(correlation_matrix, title='Correlation Heatmap of Vehicle Features')
    visualizations.append(fig2)

    # Visualization 3: Engine size vs CO2 emissions using Plotly
    fig3 = px.scatter(data, x='engine_size_liters', y='co2_emissions_gPERkm', color='fuel', 
                      title='Engine Size vs CO2 Emissions by Fuel Type', labels={'engine_size_liters': 'Engine Size (Liters)'})
    visualizations.append(fig3)

    # Visualization 4: Boxplot of CO2 emissions by transmission type using Plotly
    fig4 = px.box(data, x='transmission', y='co2_emissions_gPERkm', title='CO2 Emissions by Transmission Type')
    visualizations.append(fig4)

    return visualizations

# Call the function to create Plotly visualizations
plotly_visualizations = create_visualizations()

# Display the Plotly visualizations
for fig in plotly_visualizations:
    fig.show()

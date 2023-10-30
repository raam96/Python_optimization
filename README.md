# Optimization with Differential Evolution

This Python code utilizes the `differential_evolution` optimization algorithm from the SciPy library to solve a complex optimization problem. The primary objective is to optimize a microgrid's energy supply strategy.

## Description

The code reads input data from a CSV file (named 'input.csv') containing information on power generation from various sources, energy loads, and system parameters. The optimization aims to determine the optimal battery charge/discharge rates, photovoltaic area, and battery capacity that minimize the Levelized Cost of Electricity (LCOE) while considering various constraints.

## Algorithm

The key optimization algorithm employed in this code is Differential Evolution. It's a powerful and versatile algorithm for global optimization that works well in high-dimensional spaces.

## Objective Function

The code defines a custom objective function, `objective(u)`, which calculates the LCOE of the microgrid system. The LCOE is a critical metric for evaluating the economic viability of energy systems. The objective function incorporates a range of parameters and constraints related to battery charge/discharge, photovoltaic generation, and system dynamics. 

## Results

Upon optimization, the code provides optimal values for:

- Battery charge and discharge rates
- Photovoltaic area
- Maximum battery capacity
- Renewable and diesel power generation
- Battery state of charge
- Power supply from the microgrid

## Usage

You can utilize this code to optimize energy supply strategies for microgrids or similar systems by providing your input data in a CSV file format. Make sure to customize the parameters and constraints in the code as needed for your specific application.

## Dependencies

This code depends on the following Python libraries:

- NumPy
- SciPy

Ensure that you have these libraries installed in your Python environment.

## License

This code is provided for educational and research purposes. You are encouraged to use and modify it for your specific needs. However, please consider licensing and attribution if you plan to use it for commercial or public purposes.

**USE AT YOUR OWN RISK**.

## Author

This code was authored by the project owner.

Feel free to contact the author for any inquiries or additional information.

For detailed information on the mathematical model, constraints, and underlying principles, please refer to the code comments.

Happy optimizing!

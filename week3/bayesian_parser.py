import re
import ast
import csv

def logs_to_csv(input_file, output_csv):
    # Regex to capture Trial ID, Value, and the Parameter dictionary string
    pattern = re.compile(r"Trial (\d+) finished with value: ([\d.]+) and parameters: ({.*?})")
    
    all_data = []
    param_keys = set()

    # First, we extract all records
    with open(input_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                trial_id = match.group(1)
                value = match.group(2)
                params = ast.literal_eval(match.group(3))
                
                # Keep track of all unique parameter names for the header
                param_keys.update(params.keys())
                
                # Create a flat dictionary for the CSV row
                row = {'trial': trial_id, 'value': value}
                #unpack parameters into the row
                for k, v in params.items():
                    row[k] = v
                
                all_data.append(row)

    # Define the final header order
    fieldnames = ['trial', 'value'] + sorted(list(param_keys))

    # Writing to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Successfully saved {len(all_data)} trials to {output_csv}")

# Execution
logs_to_csv('logs/optimize_neuflow_105658.out.log', 'results/optimization_results_neuflow.csv')
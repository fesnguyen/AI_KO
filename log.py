import os
import csv

def evaluated(message, accuracy=0, model=None, filename='evaluated.csv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = ['message', 'accuracy', 'model']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If file didn't exist, write the header first
        if not file_exists:
            writer.writeheader()
        
        # Write the new row
        writer.writerow({'message': message, 'accuracy': accuracy, 'model': model})
import csv

# Function to generate SQL insert script from CSV data
def generate_insert_script(csv_filename, table_name):
    insert_statements = []
    # Open the CSV file and read data
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        
        for row in reader:
            # Format each row into an SQL insert statement
            insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join([repr(item) for item in row])});"
            insert_statements.append(insert_query)
    
    return insert_statements

# Generate insert script for bbps_txn_report
csv_filename = 'bbps_txn_report.csv'
table_name = 'bbps_txn_report'
insert_queries = generate_insert_script(csv_filename, table_name)

# Output the SQL insert statements into a file
with open('insert_bbps_txn_report.sql', mode='w') as file:
    for query in insert_queries:
        file.write(query + '\n')

print("SQL Insert script has been generated and saved to insert_bbps_txn_report.sql")

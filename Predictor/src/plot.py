import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bbps_transactions():
    # Construct the path to the CSV file
    csv_path = os.path.join('data', 'bbps_fetch_txn_report.csv')
    
    # Read the generated CSV file
    df = pd.read_csv(csv_path)

    # Create a figure with two subplots
    plt.figure(figsize=(20,10))

    # Transaction Count by Category
    plt.subplot(1,2,1)
    category_counts = df['blr_category'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Transaction Count by Category', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Transaction Value by Category
    plt.subplot(1,2,2)
    category_values = df.groupby('blr_category')['txn_amount'].sum()
    sns.barplot(x=category_values.index, y=category_values.values)
    plt.title('Total Transaction Value by Category', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Transaction Amount', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save the plot
    output_path = os.path.join('outputs', 'bbps_transaction_analysis.png')
    plt.savefig(output_path)
    plt.close()

    # Print analysis details
    print("Transaction Count by Category:")
    print(category_counts)
    print("\nTotal Transaction Value by Category:")
    print(category_values)
    print("\nAverage Transaction Amount by Category:")
    print(df.groupby('blr_category')['txn_amount'].mean())

if __name__ == '__main__':
    plot_bbps_transactions()
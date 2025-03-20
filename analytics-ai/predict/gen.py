import random
import string
from datetime import datetime, timedelta
import csv

# Helper function to generate random values for bbps_fetch_txn_report table
predefined_map = {
    "PU51": "SURYOOOOONATLE",
    "I021": "TNEB00000IND01",
    "EU51": "HPCLO0000NAT01",
    "SB51": "ANDH00000ANPAO",
    "AX91": "IIFLOOOOONATD9",
    "SB51": "SBIC00000NATDN",  
    "KM11": "KOTA0000ONATED",
    "EU51": "DISH00000NATO1",  
    "EU51": "VIDEOCONONATO1", 
    "BA01": "MUTHOOOOONATBO"
}
category_map = {
    "SURYOOOOONATLE": "Water",
    "TNEB00000IND01": "Electicity",
    "HPCLO0000NAT01": "Gas",
    "ANDH00000ANPAO": "Taxes",
    "IIFLOOOOONATD9": "IIFLOOOOONATD9",
    "SBIC00000NATDN": "Credit Card",  
    "KOTA0000ONATED": "Loan",
    "DISH00000NATO1": "DTH",  
    "VIDEOCONONATO1": "Mobile", 
    "MUTHOOOOONATBO": "Utility"
}
predefined_compliance = {
    "BOU001": "Send failed to BOU",
    "BOU002": "Timeout at BOU",
    "BOU003": "Connect Timeout to BOU",
    "BOU004": "Read Timeout to BOU",
    "BOU005": "Unable to connect to BOU"
}

# Function to generate random data for bbps_fetch_txn_report
def generate_random_data_bbps():
    bou_id_temp = random.choice(list(predefined_map.keys()))
    ref_id = f"REF{generate_random_alphanumeric(35)}".upper()
    txn_type = "PAYMENT"
    msg_id = f"MSG{generate_random_alphanumeric(35)}".upper()
    mti = "PAYMENT"
    blr_category = category_map.get(predefined_map.get(bou_id_temp))
    response_code = random.choice(["000", "200", "001"])
    payment_channel = random.choice(["Internet", "Mobile", "POS", "ATM", "Agent", "Branch"])
    cou_id = random.choice(["PP01", "IC01", "BD01", "EU01","MK01","KV01","HD01","UJ01","GP01","JH01"])
    bou_id = bou_id_temp
    bou_status = "SUCCESS" if response_code=="000" else "FAILURE"
    cust_mobile_num = f"9{random.randint(100000000, 999999999)}"
    tran_ref_id = cou_id + f"TRN{generate_random_alphanumeric(8)}".upper()
    blr_id = predefined_map.get(bou_id)
    agent_id = cou_id+ f"AGT{random.randint(1000, 9999)}"
    last_upd_host = "127.0.0.1"
    last_upd_port = "9090"
    last_upd_site_cd = f"SITE{random.randint(0, 1)}"
    random_days = random.randint(1, 180)
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    crtn_ts = (datetime.now() - timedelta(days=random_days)- timedelta(hours=random_hour, minutes=random_minute, seconds=random_second)).strftime('%Y-%m-%d %H:%M:%S')
    settlement_cycle_id = f"SCYCLE{random.randint(100, 999)}"
    complaince_cd = ""
    complaince_reason = ""
    if response_code == "001":
        complaince_cd = random.choice(list(predefined_compliance.keys()))
        complaince_reason = predefined_compliance.get(complaince_cd)
    if response_code == "200":
        complaince_cd = "BPR005"
        complaince_reason = "Payment cannot be processed"
    mandatory_cust_params = ""
    initiating_ai = cou_id
    txn_amount = f"{round(random.uniform(10, 1000), 2)}"
    on_us = random.choice(["Y", "N"])
    payment_mode = random.choice(["UPI", "Debit_Card", "Credit_Card", "Net_Banking", "Wallet", "Cash","IMPS","CBDC","AEPS"])
    status = "SUCCESS" if response_code=="000" else "FAILURE"
    return (ref_id, txn_type, msg_id, mti, blr_category, response_code, payment_channel, cou_id, bou_id,
            bou_status, cust_mobile_num, tran_ref_id, blr_id, agent_id, last_upd_host, last_upd_port,
            last_upd_site_cd, crtn_ts, settlement_cycle_id, complaince_cd, complaince_reason,
            mandatory_cust_params, initiating_ai, txn_amount, on_us, payment_mode, status)

# Helper function to generate random alphanumeric string
def generate_random_alphanumeric(length=35):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Generate 10000 rows of data and write to CSV file
header = [
    "ref_id", "txn_type", "msg_id", "mti", "blr_category", "response_code", "payment_channel", "cou_id", "bou_id",
    "bou_status", "cust_mobile_num", "tran_ref_id", "blr_id", "agent_id", "last_upd_host", "last_upd_port",
    "last_upd_site_cd", "crtn_ts", "settlement_cycle_id", "complaince_cd", "complaince_reason", 
    "mandatory_cust_params", "initiating_ai", "txn_amount", "on_us", "payment_mode","status"
]

# Writing the data to a CSV file
with open("bbps_txn_report.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    
    for _ in range(100000):  # Generate 10,000 records
        data = generate_random_data_bbps()
        writer.writerow(data)

print("10000 rows of data have been generated and saved to bbps_txn_report.csv")

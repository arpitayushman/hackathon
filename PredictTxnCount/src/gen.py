import random
import string
from datetime import datetime, timedelta
import csv
import numpy as np

# Keeping predefined mappings as requested
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
    "BA01": "MUTHOOOOONATBO",
    "FD03": "FDER00000NAT01",
}
category_map = {
    "SURYOOOOONATLE": "Water",
    "TNEB00000IND01": "Electricity",
    "HPCLO0000NAT01": "Gas",
    "ANDH00000ANPAO": "Taxes",
    "IIFLOOOOONATD9": "Educational",
    "SBIC00000NATDN": "Credit Card",  
    "KOTA0000ONATED": "Loan",
    "DISH00000NATO1": "DTH",  
    "VIDEOCONONATO1": "MobilePostpaid", 
    "MUTHOOOOONATBO": "FastTag",
    "FDER00000NAT01": "Insurance"
}
predefined_compliance = {
    "BOU001": "Send failed to BOU",
    "BOU002": "Timeout at BOU",
    "BOU003": "Connect Timeout to BOU",
    "BOU004": "Read Timeout to BOU",
    "BOU005": "Unable to connect to BOU"
}

# Enhanced realistic amount ranges and distributions
CATEGORY_AMOUNT_RANGES = {
    "Water": {"min": 100, "max": 2000, "avg": 500, "std": 200},
    "Electricity": {"min": 5, "max": 5000, "avg": 1500, "std": 500},
    "Gas": {"min": 20, "max": 3000, "avg": 800, "std": 300},
    "Taxes": {"min": 1000, "max": 10000, "avg": 3000, "std": 1000},
    "Educational": {"min": 5000, "max": 50000, "avg": 15000, "std": 5000},
    "Credit Card": {"min": 1000, "max": 25000, "avg": 5000, "std": 2000},
    "Loan": {"min": 5000, "max": 100000, "avg": 25000, "std": 10000},
    "DTH": {"min": 200, "max": 2000, "avg": 500, "std": 200},
    "MobilePostpaid": {"min": 300, "max": 3000, "avg": 800, "std": 300},
    "FastTag": {"min": 1500, "max": 5000, "avg": 1500, "std": 500},
    "Insurance": {"min": 1000, "max": 50000, "avg": 10000, "std": 5000}
}

# Enhanced payment channel probabilities per category
CATEGORY_PAYMENT_CHANNELS = {
    "Water": {"Internet": 0.4, "Mobile": 0.4, "Branch": 0.2},
    "Electricity": {"Internet": 0.5, "Mobile": 0.3, "Branch": 0.2},
    "Gas": {"Internet": 0.3, "Mobile": 0.5, "Branch": 0.2},
    "Taxes": {"Internet": 0.6, "Mobile": 0.2, "Branch": 0.2},
    "Educational": {"Internet": 0.4, "Mobile": 0.3, "Branch": 0.3},
    "Credit Card": {"Internet": 0.7, "Mobile": 0.2, "Branch": 0.1},
    "Loan": {"Internet": 0.5, "Mobile": 0.3, "Branch": 0.2},
    "DTH": {"Internet": 0.4, "Mobile": 0.5, "Branch": 0.1},
    "MobilePostpaid": {"Internet": 0.3, "Mobile": 0.6, "Branch": 0.1},
    "FastTag": {"Internet": 0.5, "Mobile": 0.4, "Branch": 0.1},
    "Insurance": {"Internet": 0.6, "Mobile": 0.3, "Branch": 0.1}
}

# Seasonal and monthly payment patterns
def get_seasonal_multiplier(category, timestamp):
    month = timestamp.month
    
    # Seasonal variations
    seasonal_multipliers = {
        "Electricity": {
            "summer_months": [5, 6, 7, 8],  # Peak summer months
            "winter_months": [12, 1, 2],     # Peak winter months
        },
        "Gas": {
            "summer_months": [12, 1, 2],     # Higher in winter
            "winter_months": [5, 6, 7, 8],   # Lower in summer
        },
        "Water": {
            "summer_months": [5, 6, 7, 8],   # Higher in summer
            "winter_months": [12, 1, 2],     # Lower in winter
        }
    }
    
    if category in seasonal_multipliers:
        if month in seasonal_multipliers[category]["summer_months"]:
            return random.uniform(1.2, 1.5)  # 20-50% higher
        elif month in seasonal_multipliers[category]["winter_months"]:
            return random.uniform(1.3, 1.6)  # 30-60% higher
    
    return 1.0

def get_monthly_cycle_multiplier(category, timestamp):
    day = timestamp.day
    
    # Monthly cycle variations
    monthly_cycle_categories = {
        "Electricity": {"start_peak": (1, 5), "end_peak": (25, 30)},
        "Taxes": {"start_peak": (1, 10), "end_peak": None},
        "Loan": {"start_peak": (1, 5), "end_peak": None},
        "Insurance": {"start_peak": None, "end_peak": (25, 30)},
        "Credit Card": {"start_peak": None, "end_peak": (25, 30)}
    }
    
    if category in monthly_cycle_categories:
        config = monthly_cycle_categories[category]
        
        if config["start_peak"] and day >= config["start_peak"][0] and day <= config["start_peak"][1]:
            return random.uniform(1.3, 1.7)  # 30-70% higher at start of month
        
        if config["end_peak"] and day >= config["end_peak"][0] and day <= config["end_peak"][1]:
            return random.uniform(1.2, 1.5)  # 20-50% higher at end of month
    
    return 1.0

def generate_random_amount(category, timestamp):
    config = CATEGORY_AMOUNT_RANGES[category]
    seasonal_multiplier = get_seasonal_multiplier(category, timestamp)
    monthly_cycle_multiplier = get_monthly_cycle_multiplier(category, timestamp)
    
    amount = random.gauss(config["avg"], config["std"])
    amount = max(config["min"], min(amount, config["max"]))
    
    return round(amount * seasonal_multiplier * monthly_cycle_multiplier, 2)

def generate_random_timestamp():
    # Generate timestamp within last 120 days and next 30 days
    days_delta = random.randint(-120, 30)
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    
    return datetime.now() + timedelta(days=days_delta, 
                                      hours=random_hour, 
                                      minutes=random_minute, 
                                      seconds=random_second)

def select_payment_channel(category):
    return random.choices(
        list(CATEGORY_PAYMENT_CHANNELS[category].keys()), 
        weights=list(CATEGORY_PAYMENT_CHANNELS[category].values())
    )[0]

# Rest of the function remains similar to previous implementation
def generate_random_data_bbps():
    bou_id_temp = random.choice(list(predefined_map.keys()))
    ref_id = f"REF{generate_random_alphanumeric(35)}".upper()
    txn_type = "PAYMENT"
    msg_id = f"MSG{generate_random_alphanumeric(35)}".upper()
    mti = "PAYMENT"
    blr_category = category_map.get(predefined_map.get(bou_id_temp))
    response_code = random.choice(["000", "200", "001"])
    
    # Timestamp generation with realistic patterns
    crtn_ts = generate_random_timestamp()
    
    # Amount generation with seasonal and category-specific variations
    txn_amount = generate_random_amount(blr_category, crtn_ts)
    
    # Payment channel selection based on category
    payment_channel = select_payment_channel(blr_category)
    
    cou_id = random.choice(["PP01", "IC01", "BD01", "EU01","MK01","KV01","HD01","UJ01","GP01","JH01"])
    bou_id = bou_id_temp
    bou_status = response_code=="000" and "SUCCESS" or "FAILURE"
    cust_mobile_num = f"9{random.randint(100000000, 999999999)}"
    tran_ref_id = cou_id + f"TRN{generate_random_alphanumeric(8)}".upper()
    blr_id = predefined_map.get(bou_id)
    agent_id = cou_id+ f"AGT{random.randint(1000, 9999)}"
    last_upd_host = "127.0.0.1"
    last_upd_port = "9090"
    last_upd_site_cd = f"SITE{random.randint(0, 1)}"
    
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
    on_us = random.choice(["Y", "N"])
    payment_mode = random.choice(["UPI", "Debit_Card", "Credit_Card", "Net_Banking", "Wallet", "Cash","IMPS","CBDC","AEPS"])
    status = response_code=="000" and "SUCCESS" or "FAILURE"
    
    # Format timestamp to string for CSV
    crtn_ts_str = crtn_ts.strftime('%Y-%m-%d %H:%M:%S')
    
    return (ref_id, txn_type, msg_id, mti, blr_category, response_code, payment_channel, cou_id, bou_id,
            bou_status, cust_mobile_num, tran_ref_id, blr_id, agent_id, last_upd_host, last_upd_port,
            last_upd_site_cd, crtn_ts_str, settlement_cycle_id, complaince_cd, complaince_reason,
            mandatory_cust_params, initiating_ai, str(txn_amount), on_us, payment_mode, status)

# Helper function to generate random alphanumeric string
def generate_random_alphanumeric(length=35):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Generating and writing data
header = [
    "ref_id", "txn_type", "msg_id", "mti", "blr_category", "response_code", "payment_channel", "cou_id", "bou_id",
    "bou_status", "cust_mobile_num", "tran_ref_id", "blr_id", "agent_id", "last_upd_host", "last_upd_port",
    "last_upd_site_cd", "crtn_ts", "settlement_cycle_id", "complaince_cd", "complaince_reason", 
    "mandatory_cust_params", "initiating_ai", "txn_amount", "on_us", "payment_mode","status"
]

# Writing the data to a CSV file
with open("bbps_fetch_txn_report.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    
    for _ in range(100000):  # Generate 100,000 records
        data = generate_random_data_bbps()
        writer.writerow(data)

print("100,000 rows of data have been generated and saved to bbps_fetch_txn_report.csv")
import random

def read_hardware_data():
    return {
        "spo2": round(random.uniform(94, 98), 1),
        "heart_rate": random.randint(70, 95),
        "resp_rate": random.randint(14, 20)
    }
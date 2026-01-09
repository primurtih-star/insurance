from backend.core.db_service import read_patient, save_patient

def get_patient(patient_id):
    return read_patient(patient_id)

def add_patient(data):
    return save_patient(data)

# msp/utils/ehr_processor.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import json
from typing import Dict, List, Any, Optional
from datetime import datetime

def process_ehr_data(ehr_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Process FHIR R4 EHR bundle into structured patient data"""
    
    processed_data = {
        "patients": [],
        "conditions": [],
        "documents": [],
        "imaging": [],
        "media": []
    }
    
    if "entry" not in ehr_bundle:
        return processed_data
    
    for entry in ehr_bundle["entry"]:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        
        if resource_type == "Patient":
            processed_data["patients"].append(process_patient(resource))
        elif resource_type == "Condition": 
            processed_data["conditions"].append(process_condition(resource))
        elif resource_type == "DocumentReference":
            processed_data["documents"].append(process_document(resource))
        elif resource_type == "ImagingStudy":
            processed_data["imaging"].append(process_imaging(resource))
        elif resource_type == "Media":
            processed_data["media"].append(process_media(resource))
    
    return processed_data

def process_patient(patient_resource: Dict) -> Dict[str, Any]:
    """Process FHIR Patient resource"""
    
    # Extract name
    name_data = patient_resource.get("name", [{}])[0]
    full_name = f"{' '.join(name_data.get('given', []))} {name_data.get('family', '')}"
    
    # Extract address
    address_data = patient_resource.get("address", [{}])[0]
    address = f"{', '.join(address_data.get('line', []))}, {address_data.get('city', '')}"
    
    # Calculate age
    birth_date = patient_resource.get("birthDate")
    age = None
    if birth_date:
        birth_year = int(birth_date.split("-")[0])
        current_year = datetime.now().year
        age = current_year - birth_year
    
    return {
        "id": patient_resource.get("id"),
        "nhs_number": extract_nhs_number(patient_resource),
        "name": full_name.strip(),
        "gender": patient_resource.get("gender"),
        "age": age,
        "birth_date": birth_date,
        "address": address,
        "phone": extract_phone(patient_resource)
    }

def process_condition(condition_resource: Dict) -> Dict[str, Any]:
    """Process FHIR Condition resource"""
    
    coding = condition_resource.get("code", {}).get("coding", [{}])[0]
    
    return {
        "id": condition_resource.get("id"),
        "patient_ref": condition_resource.get("subject", {}).get("reference"),
        "condition_name": coding.get("display"),
        "snomed_code": coding.get("code"),
        "status": condition_resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
        "system": coding.get("system")
    }

def process_document(document_resource: Dict) -> Dict[str, Any]:
    """Process FHIR DocumentReference resource"""
    
    content = document_resource.get("content", [{}])[0]
    attachment = content.get("attachment", {})
    
    return {
        "id": document_resource.get("id"),
        "patient_ref": document_resource.get("subject", {}).get("reference"),
        "document_type": document_resource.get("type", {}).get("coding", [{}])[0].get("display"),
        "title": attachment.get("title"),
        "content_type": attachment.get("contentType"),
        "url": attachment.get("url"),
        "size": attachment.get("size"),
        "date": document_resource.get("date")
    }

def process_imaging(imaging_resource: Dict) -> Dict[str, Any]:
    """Process FHIR ImagingStudy resource"""
    
    series = imaging_resource.get("series", [{}])[0]
    instance = series.get("instance", [{}])[0]
    
    return {
        "id": imaging_resource.get("id"),
        "patient_ref": imaging_resource.get("subject", {}).get("reference"),
        "modality": series.get("modality", {}).get("code"),
        "study_date": imaging_resource.get("started"),
        "title": instance.get("title"),
        "series_uid": series.get("uid"),
        "instance_uid": instance.get("uid")
    }

def process_media(media_resource: Dict) -> Dict[str, Any]:
    """Process FHIR Media resource"""
    
    content = media_resource.get("content", {})
    
    return {
        "id": media_resource.get("id"),
        "patient_ref": media_resource.get("subject", {}).get("reference"),
        "media_type": media_resource.get("type", {}).get("coding", [{}])[0].get("code"),
        "title": content.get("title"),
        "content_type": content.get("contentType"),
        "url": content.get("url"),
        "size": content.get("size"),
        "created": media_resource.get("createdDateTime")
    }

def extract_nhs_number(patient_resource: Dict) -> Optional[str]:
    """Extract NHS number from patient identifiers"""
    
    identifiers = patient_resource.get("identifier", [])
    for identifier in identifiers:
        if "nhs-number" in identifier.get("system", ""):
            return identifier.get("value")
    return None

def extract_phone(patient_resource: Dict) -> Optional[str]:
    """Extract phone number from patient telecom"""
    
    telecoms = patient_resource.get("telecom", [])
    for telecom in telecoms:
        if telecom.get("system") == "phone":
            return telecom.get("value")
    return None

def get_patient_conditions(ehr_data: Dict, patient_id: str) -> List[Dict]:
    """Get all conditions for a specific patient"""
    
    patient_ref = f"Patient/{patient_id}"
    conditions = []
    
    for condition in ehr_data.get("conditions", []):
        if condition.get("patient_ref") == patient_ref:
            conditions.append(condition)
    
    return conditions

def get_patient_documents(ehr_data: Dict, patient_id: str) -> List[Dict]:
    """Get all documents for a specific patient"""
    
    patient_ref = f"Patient/{patient_id}"
    documents = []
    
    for doc in ehr_data.get("documents", []):
        if doc.get("patient_ref") == patient_ref:
            documents.append(doc)
    
    return documents

"""Registry entries for the six corpora audited in PROTOCOL.md 9.4 (acl2023_pii_audit.tex).

Converted by scripts/prepare_pii_datasets.py into anonymous-org/pii-* (private) with the same
caps, offset check and 20-mention label floor as the four case-study corpora.  Each schema
below is exactly the set of labels that survived conversion (config/pii_dataset_stats.json);
a label listed here but absent from the data would be predicted and never matched.

Descriptions are short paraphrases of the label names, written once for these corpora and
used unchanged by every run (they are the schema the zero-shot model reads).
"""
from typing import Any, Dict

_SPLITS = {"train_split": "train", "val_splits": ["validation"], "test_splits": ["test", "validation"]}

AI4P_300K = {
    "BOD": "Person's date of birth",
    "BUILDING": "Building or house number",
    # CARDISSUER: 15 train mentions after the 50k cap, below the 20-mention floor -> dropped
    "CITY": "City or town name",
    "COUNTRY": "Country name",
    "DATE": "Calendar date",
    "DRIVERLICENSE": "Driver's license number",
    "EMAIL": "Electronic mail address",
    "GEOCOORD": "Geographic coordinates",
    "GIVENNAME1": "Person's first given name",
    "GIVENNAME2": "Person's second given name",
    "IDCARD": "Identity card number",
    "IP": "IP address",
    "LASTNAME1": "Person's first family name",
    "LASTNAME2": "Person's second family name",
    "LASTNAME3": "Person's third family name",
    "PASS": "Account password",
    "PASSPORT": "Passport number",
    "POSTCODE": "Postal or ZIP code",
    "SECADDRESS": "Secondary address unit such as an apartment or suite",
    "SEX": "Person's sex or gender",
    "SOCIALNUMBER": "Social security or national insurance number",
    "STATE": "State, province or region name",
    "STREET": "Street name",
    "TEL": "Telephone number",
    "TIME": "Time of day",
    "TITLE": "Honorific or professional title",
    "USERNAME": "Online account username",
}

AI4P_400K = {
    "ACCOUNTNUM": "Bank account number",
    "BUILDINGNUM": "Building or house number",
    "CITY": "City or town name",
    "CREDITCARDNUMBER": "Payment card number",
    "DATEOFBIRTH": "Person's date of birth",
    "DRIVERLICENSENUM": "Driver's license number",
    "EMAIL": "Electronic mail address",
    "GIVENNAME": "Person's given name",
    "IDCARDNUM": "Identity card number",
    "PASSWORD": "Account password",
    "SOCIALNUM": "Social security number",
    "STREET": "Street name",
    "SURNAME": "Person's family name",
    "TAXNUM": "Tax identification number",
    "TELEPHONENUM": "Telephone number",
    "USERNAME": "Online account username",
    "ZIPCODE": "Postal or ZIP code",
}

GRETEL_FINANCE = {
    "ACCOUNT_PIN": "Account PIN code",
    "API_KEY": "API authentication key",
    "BANK_ROUTING_NUMBER": "Bank routing number",
    "BBAN": "Basic bank account number",
    "COMPANY": "Company or organization name",
    "CREDIT_CARD_NUMBER": "Payment card number",
    "CREDIT_CARD_SECURITY_CODE": "Card security code",
    "CUSTOMER_ID": "Customer identifier",
    "DATE": "Calendar date",
    "DATE_OF_BIRTH": "Person's date of birth",
    "DATE_TIME": "Combined date and time",
    "DRIVER_LICENSE_NUMBER": "Driver's license number",
    "EMAIL": "Electronic mail address",
    "EMPLOYEE_ID": "Employee identifier",
    "FIRST_NAME": "Person's given name",
    "IBAN": "International bank account number",
    "IPV4": "IPv4 internet address",
    "IPV6": "IPv6 internet address",
    "LAST_NAME": "Person's family name",
    "LOCAL_LATLNG": "Geographic latitude and longitude",
    "NAME": "Person's full name",
    "PASSPORT_NUMBER": "Passport number",
    "PASSWORD": "Account password",
    "PHONE_NUMBER": "Telephone number",
    "SSN": "Social security number",
    "STREET_ADDRESS": "Street address",
    "SWIFT_BIC_CODE": "Bank SWIFT or BIC code",
    "TIME": "Time of day",
    "USER_NAME": "Online account username",
}

NEMOTRON_PII = {
    "ACCOUNT_NUMBER": "Bank account number",
    "AGE": "Person's age",
    "API_KEY": "API authentication key",
    "BANK_ROUTING_NUMBER": "Bank routing number",
    "BIOMETRIC_IDENTIFIER": "Biometric identifier",
    "BLOOD_TYPE": "Blood type",
    "CERTIFICATE_LICENSE_NUMBER": "Certificate or license number",
    "CITY": "City or town name",
    "COMPANY_NAME": "Company name",
    "COORDINATE": "Geographic coordinates",
    "COUNTRY": "Country name",
    "COUNTY": "County name",
    "CREDIT_DEBIT_CARD": "Credit or debit card number",
    "CUSTOMER_ID": "Customer identifier",
    "CVV": "Card security code",
    "DATE": "Calendar date",
    "DATE_OF_BIRTH": "Person's date of birth",
    "DATE_TIME": "Combined date and time",
    "DEVICE_IDENTIFIER": "Device identifier",
    "EDUCATION_LEVEL": "Education level",
    "EMAIL": "Electronic mail address",
    "EMPLOYEE_ID": "Employee identifier",
    "EMPLOYMENT_STATUS": "Employment status",
    "FAX_NUMBER": "Fax number",
    "FIRST_NAME": "Person's given name",
    "GENDER": "Person's gender",
    "HEALTH_PLAN_BENEFICIARY_NUMBER": "Health plan beneficiary number",
    "HTTP_COOKIE": "HTTP cookie value",
    "IPV4": "IPv4 internet address",
    "IPV6": "IPv6 internet address",
    "LANGUAGE": "Language a person speaks",
    "LAST_NAME": "Person's family name",
    "LICENSE_PLATE": "Vehicle registration plate",
    "MAC_ADDRESS": "Network interface MAC address",
    "MEDICAL_RECORD_NUMBER": "Medical record number",
    "NATIONAL_ID": "National identification number",
    "OCCUPATION": "Occupation or job title",
    "PASSWORD": "Account password",
    "PHONE_NUMBER": "Telephone number",
    "PIN": "Personal identification number",
    "POLITICAL_VIEW": "Political view or affiliation",
    "POSTCODE": "Postal or ZIP code",
    "RACE_ETHNICITY": "Race or ethnicity",
    "RELIGIOUS_BELIEF": "Religious belief",
    "SEXUALITY": "Sexual orientation",
    "SSN": "Social security number",
    "STATE": "State or region name",
    "STREET_ADDRESS": "Street address",
    "SWIFT_BIC": "Bank SWIFT or BIC code",
    "TAX_ID": "Tax identification number",
    "TIME": "Time of day",
    "UNIQUE_ID": "Unique identifier",
    "URL": "Web address or link",
    "USER_NAME": "Online account username",
    "VEHICLE_IDENTIFIER": "Vehicle identifier such as a VIN",
}

TAB = {
    "PERSON": "Person name",
    "CODE": "Identification number or code, such as an application or case number",
    "LOC": "Location or place",
    "ORG": "Organization, institution or company",
    "DEM": "Demographic attribute such as nationality, ethnicity, profession or age",
    "DATETIME": "Date, time or duration",
    "QUANTITY": "Quantity or amount, such as money or a percentage",
    "MISC": "Other identifying information",
}

MAPA = {
    "ADDRESS": "Postal address or place",
    "AMOUNT": "Monetary amount or quantity",
    "DATE": "Calendar date",
    "ORGANISATION": "Organisation or institution name",
    "PERSON": "Person name",
}


def _entry(repo: str, descriptions: Dict[str, str], tag: str, **overrides: Any) -> Dict[str, Any]:
    return {"repo": repo, "label_descriptions": descriptions, **_SPLITS, "tag": tag, **overrides}


AUDIT_DATASETS: Dict[str, Dict[str, Any]] = {
    "ai4p_300k": _entry("anonymous-org/pii-ai4p-300k", AI4P_300K, "ai4p_300k"),
    "ai4p_400k": _entry("anonymous-org/pii-ai4p-400k", AI4P_400K, "ai4p_400k"),
    "gretel_finance": _entry("anonymous-org/pii-gretel-finance", GRETEL_FINANCE, "gretel_finance"),
    "nemotron_pii": _entry("anonymous-org/pii-nemotron", NEMOTRON_PII, "nemotron_pii"),
    "tab": _entry("anonymous-org/pii-tab", TAB, "tab"),
    # training copy of TAB: train/validation cut into <=1000-word pieces without cutting a
    # span (scripts/chunk_tab.py, PROTOCOL.md 9.15); test identical to anonymous-org/pii-tab
    "tab_chunked": _entry("anonymous-org/pii-tab-chunked", TAB, "tab"),
    # test-only release
    "mapa": _entry("anonymous-org/pii-mapa", MAPA, "mapa", train_split=None, val_splits=[],
                   test_splits=["test"]),
}

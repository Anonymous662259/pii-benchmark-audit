from typing import Dict

# PII taxonomy — 44 classes (native GLiNER2 schema).
# Source: PII Taxonomy doc (Identity / Contact & Location / Digital & Network /
# Financial / Employment / Government & Legal / Time / Sensitive).
# COMPANY_NAME is intentionally excluded (folded into ORGANIZATION).
PII_LABEL_DESCRIPTIONS: Dict[str, str] = {
 "PERSON": "Person names",
    "DATE": "Calendar dates",
    "JOB_TITLE": "Job titles",
    "TIME": "Time values",
    "LOCATION": "General locations",
    "PREFIX": "Titles before names",
    "ORGANIZATION": "Organization names",
    "EMAIL": "Email addresses",
    "PHONE": "Phone numbers",
    "ADDRESS": "Street addresses",
    "GENDER": "Sex or gender",
    "COORDINATE": "Geographic coordinates",
    "MONEY": "Monetary amounts",
    "IP": "IP addresses",
    "ZIP_CODE": "Postal or ZIP codes",
    "NATIONALITY": "Legal nationality",
    "AGE": "Age values",
    "USERNAME": "Usernames or handles",
    "URL": "Web links or URLs",
    "MARITAL": "Marital status",
    "TRADE_UNION": "Trade union membership",
    "BIRTHDATE": "Date of birth",
    "CARD_ISSUER": "Card brands or issuers",
    "PIN": "Authentication PINs",
    "PASSPORT": "Passport numbers",
    "MEDICAL_INFO": "Health or medical information",
    "IBAN": "International bank account numbers",
    "PASSWORD": "Account passwords",
    "ETHNICITY": "Ethnic or racial group",
    "RELIGION": "Religious affiliation",
    "EMPLOYEE_ID": "Employee identifiers",
    "INSURANCE_ID": "Insurance identifiers",
    "TIN": "Tax identification numbers",
    "SWIFT": "Bank SWIFT or BIC codes",
    "CARD_NUMBER": "Payment card numbers",
    "ACCOUNT_ID": "Account identifiers",
    "WALLET": "Crypto wallet addresses",
    "NATIONAL_ID": "National ID numbers",
    "CVV": "Card security codes",
    "BANK_ACCOUNT": "Bank account numbers",
    "LICENSE": "Driver license numbers",
    "TICKET_ID": "Support or case IDs",
    "PLATE": "Vehicle plate numbers",
    "API_KEY": "API keys or tokens",
}

# Legacy (old PII-schema) → native-taxonomy aliases. normalize_label() applies
# these so any dataset still carrying old label names is migrated on the fly.
LABEL_MAPPINGS: Dict[str, str] = {
}

ALL_LABELS_ENUM = list(PII_LABEL_DESCRIPTIONS.keys())

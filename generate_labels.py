import pandas as pd
import re
import json
import requests

# Load your cleaned dataset
df = pd.read_csv("/Users/mithra/advanced_siem_cleaned.csv")

# ------------------------
# 1. Load MITRE Techniques (STIX format)
# ------------------------
print("Downloading MITRE ATT&CK STIX mapping...")

url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
response = requests.get(url)

if response.status_code != 200:
    raise Exception("Failed to download MITRE dataset. Check internet connection.")

mitre_json = response.json()

technique_map = {}  # {technique_id: tactic_name}

for obj in mitre_json["objects"]:
    if obj.get("type") == "attack-pattern":
        # external_id looks like "T1059"
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id")
                break
        else:
            tid = None

        # tactics stored in kill_chain_phases
        if tid:
            kcp = obj.get("kill_chain_phases", [])
            if kcp:
                tactic = kcp[0]["phase_name"]  # use first tactic
                technique_map[tid] = tactic

print("Loaded MITRE technique mappings:", len(technique_map))


# -----------------------------------
# 2. Extract MITRE Technique IDs
# -----------------------------------
pattern = r"T\d{4}(?:\.\d{3})?"

def extract_technique(text):
    if pd.isna(text):
        return None
    match = re.search(pattern, str(text))
    return match.group(0) if match else None

df["technique"] = df["description"].apply(extract_technique)
df["technique"] = df["technique"].fillna(df["additional_info"].apply(extract_technique))


# -----------------------------------
# 3. Map Technique → MITRE Tactic
# -----------------------------------
def map_to_tactic(tech):
    if tech is None:
        return "benign"
    base = tech.split(".")[0]  # Remove subtechnique
    return technique_map.get(base, "benign")

df["tactic"] = df["technique"].apply(map_to_tactic)


# -----------------------------------
# 4. Map Tactic → Attack Stage
# -----------------------------------
tactic_to_stage = {
    "reconnaissance": "recon",
    "resource-development": "recon",

    "initial-access": "exploit",
    "execution": "exploit",

    "persistence": "priv_esc",
    "privilege-escalation": "priv_esc",
    "defense-evasion": "priv_esc",

    "credential-access": "lateral",
    "discovery": "lateral",
    "lateral-movement": "lateral",

    "collection": "exfil",
    "exfiltration": "exfil",
    "impact": "exfil",
}

df["attack_stage"] = df["tactic"].apply(lambda t: tactic_to_stage.get(t, "benign"))


# -----------------------------------
# 5. Convert Attack Stage → Numeric Label
# -----------------------------------
stage_to_label = {
    "benign": 0,
    "recon": 1,
    "exploit": 2,
    "priv_esc": 3,
    "lateral": 4,
    "exfil": 5,
}

df["label"] = df["attack_stage"].apply(lambda x: stage_to_label[x])


# -----------------------------------
# 6. Save final labeled dataset
# -----------------------------------
df.to_csv("advanced_siem_labeled.csv", index=False)

print("\nLabel generation complete!")
print("Saved: advanced_siem_labeled.csv")
print(df[["technique", "tactic", "attack_stage", "label"]].head())

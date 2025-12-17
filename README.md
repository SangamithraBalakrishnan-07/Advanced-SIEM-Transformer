
---

## Dataset

The project uses the **Advanced SIEM Dataset** from HuggingFace:

🔗 https://huggingface.co/datasets/darkknight25/advanced_siem_dataset

### Data files stored in your `/data` directory:

- `advanced_siem_cleaned.csv`
- `advanced_siem_labeled.csv`

These files were produced through:
1. JSON flattening  
2. Feature normalization + encoding  
3. MITRE ATT&CK technique extraction  
4. Attack-stage label generation

---
# 1. Folder Structure

```

Advanced-SIEM-Transformer/
│
│── data/
│ ├── advanced_siem_cleaned.csv
│ ├── advanced_siem_labeled.csv
│
│── logbert-cls/ # Log-BERT fine-tuned classifier
│── logbert-mlm/ # Log-BERT MLM pretraining outputs
│
│── notebooks/
│ ├── preprocess.py
│ ├── generate_labels.py
│ ├── encode_build_sequence.py
│ ├── transformer_classifier.py
│ ├── logbert_model.py
│ ├── evaluation_plots.py
│ └── attack_reconstruction.py
│
│── reconstruction_outputs/
│ ├── reconstructed_chains.json
│ ├── chain_1_graph.png
│ ├── chain_1_timeline.png
│
│── results/
│ ├── confusion_model1.png
│ ├── confusion_model2.png
│ ├── roc_model1.png
│ ├── roc_model2.png
│ ├── attention_heatmap_model2.png
│ └── model_comparison_metrics.csv
│
│── requirements.txt
│── README.md

```
---

# 2. Results (What Each Output Means)

### **confusion_model1.png**  
Confusion matrix for Transformer Encoder (Model 1).  
Shows misclassification across attack stages.

### **confusion_model2.png**  
Confusion matrix for Log-BERT — perfect classification (diagonal only).

---

### **roc_model1.png**  
ROC curve for Transformer Encoder — moderate separability.

### **roc_model2.png**  
ROC curve for Log-BERT — AUC = 1.0 (near-perfect).

---

### **attention_heatmap_model2.png**  
Shows which event tokens Log-BERT attends to.  
Highlights MITRE techniques, high-risk events, and suspicious sequences.

---

### **model_comparison_metrics.csv**  
Table comparing both models (Accuracy, Precision, Recall, F1):

| Model                | Accuracy | Precision | Recall | F1 |
|---------------------|----------|-----------|--------|----|
| Transformer Encoder | 0.7195   | 0.5176    | 0.7195 | 0.6021 |
| Log-BERT            | 1.0000   | 1.0000    | 1.0000 | 1.0000 |

---

### **Attack Reconstruction Outputs**

Located in `reconstruction_outputs/`:

- **reconstructed_chains.json**  
- **chain_1_timeline.png**  
- **chain_1_graph.png**

Generated using:
- Method C — Greedy Decoding  
- Method D — Graph-Based Correlation  

These reconstruct multi-stage attacks such as:


## Installation

Clone the repository:

```bash
git clone https://github.com/SangamithraBalakrishnan-07/Advanced-SIEM-Transformer.git
cd Advanced-SIEM-Transformer

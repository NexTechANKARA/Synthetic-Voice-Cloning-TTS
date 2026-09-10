import json

nb = json.load(open(r'c:\Users\INTEL\Downloads\GEN_AI\Trained\Trained_models\voice_cloning_models_1_2_3_2-2.ipynb', 'r', encoding='utf-8'))

with open(r'c:\Users\INTEL\Downloads\GEN_AI\notebook_extracted_code.py', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            out.write(f"\n# ==================== CELL {i} ====================\n")
            out.write(''.join(c['source']))
            out.write('\n')

print("Extracted all code cells successfully.")

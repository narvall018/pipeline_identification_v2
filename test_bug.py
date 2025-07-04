#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

def test_correct_columns():
    """Test que les colonnes contiennent les bonnes données."""
    print("🔍 Test des colonnes corrigées")
    
    # Charger le fichier
    df = pd.read_parquet("data/output/feature_matrix/features_complete.parquet")
    print(f"   ✓ {len(df)} lignes chargées")
    
    # Test 1: Vérifier qu'on a bien les 4 colonnes MS2
    expected_cols = ['ms2_mz_experimental', 'ms2_intensities_experimental', 
                    'ms2_mz_reference', 'ms2_intensities_reference']
    
    for col in expected_cols:
        if col in df.columns:
            print(f"   ✅ {col} présent")
        else:
            print(f"   ❌ {col} MANQUANT")
    
    # Test 2: Exemples avec spectres expérimentaux
    exp_data = df[df['ms2_mz_experimental'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    print(f"\n📊 {len(exp_data)} lignes avec spectres expérimentaux")
    
    if len(exp_data) > 0:
        ex = exp_data.iloc[0]
        print(f"   • Feature {ex['feature_idx']}:")
        print(f"     - Exp m/z: {ex['ms2_mz_experimental'][:3]}")
        print(f"     - Exp int: {ex['ms2_intensities_experimental'][:3]}")
    
    # Test 3: Exemples avec spectres de référence
    ref_data = df[
        (df['match_name'].notna()) & 
        (df['ms2_mz_reference'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ]
    print(f"\n📚 {len(ref_data)} lignes avec spectres de référence")
    
    if len(ref_data) > 0:
        ex = ref_data.iloc[0]
        print(f"   • {ex['match_name']}:")
        print(f"     - Ref m/z: {ex['ms2_mz_reference'][:3]}")
        print(f"     - Ref int: {ex['ms2_intensities_reference'][:3]}")
        print(f"     - Énergie: {ex.get('collision_energy_reference', 'N/A')}")
    
    # Test 4: Vérifier que les données sont différentes
    both_data = df[
        (df['ms2_mz_experimental'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df['ms2_mz_reference'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ]
    
    if len(both_data) > 0:
        ex = both_data.iloc[0]
        exp_mz = ex['ms2_mz_experimental']
        ref_mz = ex['ms2_mz_reference']
        
        print(f"\n🔍 Comparaison exp vs ref pour {ex.get('match_name', 'Feature ' + str(ex['feature_idx']))}:")
        print(f"   • Exp: {len(exp_mz)} pics, premiers: {exp_mz[:3]}")
        print(f"   • Ref: {len(ref_mz)} pics, premiers: {ref_mz[:3]}")
        print(f"   • Identiques: {exp_mz == ref_mz}")

if __name__ == "__main__":
    test_correct_columns()
#!/usr/bin/env python3
# scripts/utils/preprocess_database.py
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import time

# Ajouter le chemin du projet pour les imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabasePreprocessor:
    """Classe pour pré-traiter la base de données une seule fois."""
    
    def __init__(self):
        self.config = Config.IDENTIFICATION
        
    def _convert_peaks_string_to_list(self, peaks_data) -> list:
        """Conversion correcte des spectres MS2 qui préserve les listes valides."""
        try:
            # Si c'est déjà une liste valide, la nettoyer et la retourner
            if isinstance(peaks_data, list):
                if len(peaks_data) == 0:
                    return []
                
                # Nettoyer la liste en gardant seulement les nombres valides
                result = []
                for item in peaks_data:
                    try:
                        if pd.notna(item):  # Vérifier que ce n'est pas NaN
                            val = float(item)
                            if not (np.isnan(val) or np.isinf(val)):
                                result.append(val)
                    except (ValueError, TypeError):
                        continue
                return result
            
            # Si c'est un array numpy
            if isinstance(peaks_data, np.ndarray):
                if peaks_data.size == 0:
                    return []
                try:
                    # Convertir en liste et nettoyer
                    peaks_list = peaks_data.tolist()
                    return self._convert_peaks_string_to_list(peaks_list)
                except:
                    return []
            
            # Si c'est un scalaire NaN
            if isinstance(peaks_data, (int, float)) and pd.isna(peaks_data):
                return []
            
            # Si c'est None
            if peaks_data is None:
                return []
            
            # Si c'est une string
            if isinstance(peaks_data, str):
                peaks_str = peaks_data.strip()
                if not peaks_str or peaks_str.lower() in ['nan', 'none', 'null']:
                    return []
                
                # Enlever les crochets
                peaks_str = peaks_str.strip('[]')
                if not peaks_str:
                    return []
                
                # Essayer différents séparateurs
                for sep in [',', ';', ' ', '\t']:
                    if sep in peaks_str:
                        values = peaks_str.split(sep)
                        result = []
                        for val in values:
                            try:
                                val = val.strip()
                                if val:
                                    num_val = float(val)
                                    if not (np.isnan(num_val) or np.isinf(num_val)):
                                        result.append(num_val)
                            except (ValueError, TypeError):
                                continue
                        if result:
                            return result
                
                # Essayer comme un seul nombre
                try:
                    val = float(peaks_str)
                    if not (np.isnan(val) or np.isinf(val)):
                        return [val]
                except (ValueError, TypeError):
                    pass
                
                return []
            
            # Si c'est un nombre unique
            if isinstance(peaks_data, (int, float)):
                if not (pd.isna(peaks_data) or np.isinf(peaks_data)):
                    return [float(peaks_data)]
                return []
            
            # Type non reconnu
            return []
        
        except Exception as e:
            logger.debug(f"Erreur conversion pics: {type(peaks_data)}, {str(e)}")
            return []

    def preprocess_database(self, force_reprocess: bool = False):
        """
        Pré-traite la base de données et sauvegarde la version traitée.
        
        Args:
            force_reprocess: Si True, force le retraitement même si déjà fait
        """
        db_path = Path(Config.PATHS.INPUT_DATABASES) / self.config.database_file
        
        if not db_path.exists():
            raise FileNotFoundError(f"Base de données non trouvée: {db_path}")

        print(f"🔄 Chargement de la base de données: {db_path}")
        start_time = time.time()
        
        # Charger la base
        df = pd.read_hdf(db_path, key=self.config.database_key)
        print(f"   ✓ {len(df)} entrées chargées")
        
        # Vérifier si déjà traité (sauf si force_reprocess)
        if not force_reprocess and self._is_already_processed(df):
            print("   ℹ️ Base de données déjà pré-traitée, aucun traitement nécessaire")
            return
        
        print("\n🔧 Pré-traitement de la base de données...")
        
        # 1. Conversion des types numériques
        print("   • Conversion des types numériques...")
        dtype_map = {
            'mz': np.float32,
            'ccs_exp': np.float32,
            'ccs_pred': np.float32,
            'Observed_RT': np.float32,
            'Predicted_RT': np.float32
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # 2. Traitement des spectres MS2
        if 'peaks_ms2_mz' in df.columns and 'peaks_ms2_intensities' in df.columns:
            print("   • Traitement des spectres MS2...")
            
            # Conversion des spectres avec barre de progression
            total_rows = len(df)
            batch_size = 10000
            
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                print(f"     - Traitement lignes {i+1} à {batch_end}/{total_rows}")
                
                # Traiter le batch
                batch = df.iloc[i:batch_end]
                df.loc[i:batch_end, 'peaks_ms2_mz'] = batch['peaks_ms2_mz'].apply(
                    self._convert_peaks_string_to_list
                )
                df.loc[i:batch_end, 'peaks_ms2_intensities'] = batch['peaks_ms2_intensities'].apply(
                    self._convert_peaks_string_to_list
                )
            
            print("   • Validation des spectres MS2...")
            
            # Validation avec fonction correcte
            def validate_ms2_row(row):
                mz_data = row['peaks_ms2_mz']
                int_data = row['peaks_ms2_intensities']
                
                return (isinstance(mz_data, list) and len(mz_data) > 0 and
                        isinstance(int_data, list) and len(int_data) > 0 and
                        len(mz_data) == len(int_data))
            
            valid_ms2_mask = df.apply(validate_ms2_row, axis=1)
            valid_count = valid_ms2_mask.sum()
            
            print(f"   ✓ {valid_count}/{len(df)} entrées avec spectres MS2 valides")
            
            # Nettoyage des énergies de collision
            if 'Collision_energy' in df.columns:
                print("   • Nettoyage des énergies de collision...")
                def clean_collision_energy(energy_str):
                    try:
                        if pd.isna(energy_str):
                            return np.nan
                        
                        energy_str = str(energy_str).strip()
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', energy_str)
                        if match:
                            return float(match.group(1))
                        return np.nan
                    except:
                        return np.nan
                
                df['Collision_energy_clean'] = df['Collision_energy'].apply(clean_collision_energy)
                
                # Statistiques
                collision_stats = df[valid_ms2_mask]['Collision_energy_clean'].value_counts()
                print("   ✓ Spectres MS2 valides par énergie de collision:")
                for energy, count in collision_stats.head(10).items():
                    if pd.notna(energy):
                        print(f"      {energy} eV: {count} spectres")
        
        # 3. Pré-calculs
        print("   • Calculs supplémentaires...")
        df['Name_str'] = df['Name'].astype(str).fillna('')
        df['adduct_str'] = df['adduct'].astype(str).fillna('')
        df['molecule_id'] = df['Name_str'] + '_' + df['adduct_str']
        
        # Statut MS2 correct
        def has_valid_ms2(row):
            mz_data = row['peaks_ms2_mz']
            int_data = row['peaks_ms2_intensities']
            
            return (isinstance(mz_data, list) and len(mz_data) > 0 and
                    isinstance(int_data, list) and len(int_data) > 0 and
                    len(mz_data) == len(int_data))
        
        df['has_ms2_db'] = df.apply(has_valid_ms2, axis=1).astype(int)
        
        # 4. Marqueur de traitement
        df.attrs['preprocessed'] = True
        df.attrs['preprocessed_version'] = '1.0'
        df.attrs['preprocessed_timestamp'] = pd.Timestamp.now().isoformat()
        
        # 5. Sauvegarde
        print(f"\n💾 Sauvegarde de la base pré-traitée...")
        df.to_hdf(db_path, key=self.config.database_key, mode='w', complevel=9)
        
        ms2_count = df['has_ms2_db'].sum()
        processing_time = time.time() - start_time
        
        print(f"   ✅ Base de données pré-traitée avec succès!")
        print(f"   • {len(df)} entrées au total")
        print(f"   • {ms2_count} entrées avec MS2 valides")
        print(f"   • Temps de traitement: {processing_time:.1f} secondes")

    def _is_already_processed(self, df: pd.DataFrame) -> bool:
        """Vérifie si la base est déjà pré-traitée."""
        try:
            # Vérifier les attributs
            if hasattr(df, 'attrs') and df.attrs.get('preprocessed', False):
                return True
            
            # Vérifier si les colonnes calculées existent
            required_cols = ['Name_str', 'adduct_str', 'molecule_id', 'has_ms2_db']
            if all(col in df.columns for col in required_cols):
                # Vérifier si les spectres MS2 sont déjà des listes
                if 'peaks_ms2_mz' in df.columns:
                    sample_ms2 = df['peaks_ms2_mz'].dropna().iloc[:100]  # Échantillon
                    if len(sample_ms2) > 0 and all(isinstance(x, list) for x in sample_ms2):
                        return True
            
            return False
        except:
            return False

def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-traite la base de données MS")
    parser.add_argument('--force', action='store_true', 
                       help='Force le retraitement même si déjà fait')
    
    args = parser.parse_args()
    
    try:
        preprocessor = DatabasePreprocessor()
        preprocessor.preprocess_database(force_reprocess=args.force)
        print("\n🎉 Pré-traitement terminé avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du pré-traitement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

#scripts/processing/identification.py
#-*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.distance import cdist
from ..config.config import Config  
from ..utils.matching_utils import calculate_match_scores, assign_confidence_level 

class CompoundIdentifier:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = Config.IDENTIFICATION
        self.db = pd.DataFrame()
        # Utiliser 75% des cœurs disponibles
        self.n_workers = max(1, int(mp.cpu_count() * 0.75))
        self.load_database()

    def load_database(self) -> None:
        try:
            db_path = Path(Config.PATHS.INPUT_DATABASES) / self.config.database_file
            if not db_path.exists():
                raise FileNotFoundError(f"Base de données non trouvée: {db_path}")

            self.db = pd.read_hdf(db_path, key=self.config.database_key)
            
            # Vérifier si la base est déjà pré-traitée
            is_preprocessed = self._is_database_preprocessed()
            
            if not is_preprocessed:
                print("⚠️  Base de données non pré-traitée détectée!")
                print("   Pour optimiser les performances, exécutez d'abord:")
                print("   python scripts/utils/preprocess_database.py")
                print("\n   Traitement en cours (plus lent)...")
                self._apply_legacy_processing()
            else:
                print("   ✓ Base de données pré-traitée chargée")
            
            ms2_count = self.db['has_ms2_db'].sum()
            self.logger.info(f"Base de données chargée: {len(self.db)} entrées, {ms2_count} avec MS2")

        except Exception as e:
            self.logger.error(f"Erreur chargement base: {str(e)}")
            raise

    def _is_database_preprocessed(self) -> bool:
        """Vérifie si la base de données est déjà pré-traitée."""
        try:
            # Vérifier les attributs
            if hasattr(self.db, 'attrs') and self.db.attrs.get('preprocessed', False):
                return True
            
            # Vérifier si les colonnes calculées existent et sont cohérentes
            required_cols = ['Name_str', 'adduct_str', 'molecule_id', 'has_ms2_db']
            if not all(col in self.db.columns for col in required_cols):
                return False
            
            # Vérifier si les spectres MS2 sont déjà des listes
            if 'peaks_ms2_mz' in self.db.columns:
                sample_ms2 = self.db['peaks_ms2_mz'].dropna().iloc[:10]  # Petit échantillon
                if len(sample_ms2) > 0:
                    # Tous doivent être des listes
                    return all(isinstance(x, list) for x in sample_ms2)
            
            return True  # Si pas de MS2, considérer comme traité
            
        except Exception:
            return False

    def _apply_legacy_processing(self) -> None:
        """Applique le traitement legacy si la base n'est pas pré-traitée."""
        # Conversion des types numériques
        dtype_map = {
            'mz': np.float32,
            'ccs_exp': np.float32,
            'ccs_pred': np.float32,
            'Observed_RT': np.float32,
            'Predicted_RT': np.float32
        }
        
        for col, dtype in dtype_map.items():
            if col in self.db.columns:
                self.db[col] = self.db[col].astype(dtype)

        # Traitement des spectres MS2 si nécessaire
        if ('peaks_ms2_mz' in self.db.columns and 
            'peaks_ms2_intensities' in self.db.columns):
            
            # Vérifier si déjà traité
            sample_ms2 = self.db['peaks_ms2_mz'].dropna().iloc[:5]
            if len(sample_ms2) > 0 and not all(isinstance(x, list) for x in sample_ms2):
                print("   ✓ Traitement des spectres MS2...")
                
                # Conversion des spectres
                for col in ['peaks_ms2_mz', 'peaks_ms2_intensities']:
                    print(f"     - Nettoyage de {col}...")
                    self.db[col] = self.db[col].apply(self._convert_peaks_string_to_list)
                
                # Validation
                def validate_ms2_row(row):
                    mz_data = row['peaks_ms2_mz']
                    int_data = row['peaks_ms2_intensities']
                    
                    return (isinstance(mz_data, list) and len(mz_data) > 0 and
                            isinstance(int_data, list) and len(int_data) > 0 and
                            len(mz_data) == len(int_data))
                
                valid_ms2_mask = self.db.apply(validate_ms2_row, axis=1)
                valid_count = valid_ms2_mask.sum()
                
                print(f"   ✓ {valid_count}/{len(self.db)} entrées avec spectres MS2 valides")
                
                # Nettoyage des énergies de collision
                if 'Collision_energy' in self.db.columns:
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
                    
                    self.db['Collision_energy_clean'] = self.db['Collision_energy'].apply(clean_collision_energy)
                    
                    # Statistiques
                    collision_stats = self.db[valid_ms2_mask]['Collision_energy_clean'].value_counts()
                    print("   ✓ Spectres MS2 valides par énergie de collision:")
                    for energy, count in collision_stats.head(10).items():
                        if pd.notna(energy):
                            print(f"      {energy} eV: {count} spectres")
        
        # Pré-calculs
        self.db['Name_str'] = self.db['Name'].astype(str).fillna('')
        self.db['adduct_str'] = self.db['adduct'].astype(str).fillna('')
        self.db['molecule_id'] = self.db['Name_str'] + '_' + self.db['adduct_str']
        
        # Statut MS2
        def has_valid_ms2(row):
            mz_data = row['peaks_ms2_mz']
            int_data = row['peaks_ms2_intensities']
            
            return (isinstance(mz_data, list) and len(mz_data) > 0 and
                    isinstance(int_data, list) and len(int_data) > 0 and
                    len(mz_data) == len(int_data))
        
        self.db['has_ms2_db'] = self.db.apply(has_valid_ms2, axis=1).astype(int)

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
            self.logger.debug(f"Erreur conversion pics: {type(peaks_data)}, {str(e)}")
            return []

    def find_matches_vectorized(self, peaks_df: pd.DataFrame, tolerances: Dict[str, float]) -> pd.DataFrame:
        """Version vectorisée de la recherche des correspondances."""
        try:
            # Conversion en arrays NumPy pour optimisation
            peak_mz = peaks_df['mz'].values.astype(np.float32)
            peak_rt = peaks_df['retention_time'].values.astype(np.float32)
            peak_ccs = peaks_df['CCS'].values.astype(np.float32)
            
            db_mz = self.db['mz'].values.astype(np.float32)
            
            # Calcul vectorisé des différences de m/z
            mz_tolerance = np.outer(peak_mz, np.ones_like(db_mz)) * tolerances['mz_ppm'] * 1e-6
            mz_diff_matrix = np.abs(np.subtract.outer(peak_mz, db_mz))
            valid_mz_mask = mz_diff_matrix <= mz_tolerance
            
            all_matches = []
            
            # Traitement par lots pour éviter la surcharge mémoire
            batch_size = 1000
            for i in range(0, len(peaks_df), batch_size):
                batch_mask = valid_mz_mask[i:i+batch_size]
                batch_peaks = peaks_df.iloc[i:i+batch_size]
                
                # Pour chaque pic dans le lot
                for peak_idx, peak_matches in enumerate(batch_mask):
                    peak = batch_peaks.iloc[peak_idx]
                    matched_db_indices = np.where(peak_matches)[0]
                    
                    if len(matched_db_indices) == 0:
                        continue
                        
                    # Traitement vectorisé des correspondances
                    matched_db = self.db.iloc[matched_db_indices]
                    
                    # Calcul vectorisé des erreurs
                    rt_errors = np.full(len(matched_db), np.inf)
                    ccs_errors = np.full(len(matched_db), np.inf)
                    
                    # RT errors
                    obs_rt_mask = pd.notna(matched_db['Observed_RT'])
                    pred_rt_mask = pd.notna(matched_db['Predicted_RT'])
                    
                    if obs_rt_mask.any():
                        rt_errors[obs_rt_mask] = np.abs(
                            peak.retention_time - matched_db.loc[obs_rt_mask, 'Observed_RT']
                        )
                    if pred_rt_mask.any():
                        rt_errors[pred_rt_mask] = np.minimum(
                            rt_errors[pred_rt_mask],
                            np.abs(peak.retention_time - matched_db.loc[pred_rt_mask, 'Predicted_RT'])
                        )
                    
                    # CCS errors
                    exp_ccs_mask = pd.notna(matched_db['ccs_exp'])
                    pred_ccs_mask = pd.notna(matched_db['ccs_pred'])
                    
                    if exp_ccs_mask.any():
                        ccs_errors[exp_ccs_mask] = np.abs(
                            (peak.CCS - matched_db.loc[exp_ccs_mask, 'ccs_exp']) / 
                            matched_db.loc[exp_ccs_mask, 'ccs_exp'] * 100
                        )
                    if pred_ccs_mask.any():
                        ccs_errors[pred_ccs_mask] = np.minimum(
                            ccs_errors[pred_ccs_mask],
                            np.abs((peak.CCS - matched_db.loc[pred_ccs_mask, 'ccs_pred']) / 
                                 matched_db.loc[pred_ccs_mask, 'ccs_pred'] * 100)
                        )
                    
                    # Filtrage final
                    valid_matches = (
                        (rt_errors <= tolerances['rt_min']) | 
                        (rt_errors == np.inf)
                    ) & (
                        (ccs_errors <= tolerances['ccs_percent']) | 
                        (ccs_errors == np.inf)
                    )
                    
                    if not valid_matches.any():
                        continue
                        
                    # Création des correspondances
                    for match_idx in np.where(valid_matches)[0]:
                        match = matched_db.iloc[match_idx]
                        
                        match_details = {
                            'peak_mz': peak.mz,
                            'peak_rt': peak.retention_time,
                            'peak_dt': peak.drift_time,
                            'peak_intensity': peak.intensity,
                            'peak_ccs': peak.CCS,
                            'match_name': match.Name,
                            'match_adduct': match.adduct,
                            'match_smiles': match.SMILES,
                            'categories': match.categories,
                            'match_mz': match.mz,
                            'mz_error_ppm': (peak.mz - match.mz) / match.mz * 1e6,
                            'match_ccs_exp': match.ccs_exp,
                            'match_ccs_pred': match.ccs_pred,
                            'match_rt_obs': match.Observed_RT,
                            'match_rt_pred': match.Predicted_RT,
                            'rt_error_min': rt_errors[match_idx] if rt_errors[match_idx] != np.inf else None,
                            'ccs_error_percent': ccs_errors[match_idx] if ccs_errors[match_idx] != np.inf else None,
                            'has_ms2_db': match.has_ms2_db,
                            'molecule_id': match.molecule_id
                        }
                        
                        # Ajout des scores toxicologiques si présents
                        tox_columns = ['LC50_48_hr_ug/L', 'EC50_72_hr_ug/L', 'LC50_96_hr_ug/L']
                        for col in tox_columns:
                            if col in match and pd.notna(match[col]):
                                match_details[f"daphnia_{col}" if '48' in col else
                                            f"algae_{col}" if '72' in col else
                                            f"pimephales_{col}"] = float(match[col])
                            else:
                                match_details[f"daphnia_{col}" if '48' in col else
                                            f"algae_{col}" if '72' in col else
                                            f"pimephales_{col}"] = None
                        
                        # Calcul des scores et niveau de confiance
                        score_details = calculate_match_scores(match_details, tolerances)
                        confidence_level, confidence_reason = assign_confidence_level(match_details, tolerances)
                        
                        match_details.update({
                            'individual_scores': score_details['individual_scores'],
                            'global_score': score_details['global_score'],
                            'ccs_source': score_details['ccs_source'],
                            'rt_source': score_details['rt_source'],
                            'confidence_level': confidence_level,
                            'confidence_reason': confidence_reason
                        })
                        
                        all_matches.append(match_details)
            
            # Création du DataFrame final
            matches_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
            
            if not matches_df.empty:
                # Trie et dédoublonnage optimisés
                matches_df['sort_key'] = matches_df['has_ms2_db'].astype(str) + '_' + \
                                       matches_df['global_score'].astype(str)
                                       
                matches_df = matches_df.sort_values(
                    'sort_key', ascending=False
                ).drop_duplicates(
                    subset='molecule_id', keep='first'
                ).drop(columns=['sort_key'])
                
                matches_df = matches_df.sort_values(
                    ['confidence_level', 'global_score'], 
                    ascending=[True, False]
                )
            
            return matches_df
            
        except Exception as e:
            self.logger.error(f"Erreur dans la recherche vectorisée : {str(e)}")
            raise

    def identify_compounds(self, peaks_df: pd.DataFrame, output_dir: str) -> Optional[pd.DataFrame]:
        self.logger.info("Début du processus d'identification des composés.")
        
        try:
            # Conversion des types pour optimisation
            peaks_df['mz'] = peaks_df['mz'].astype(np.float32)
            peaks_df['retention_time'] = peaks_df['retention_time'].astype(np.float32)
            peaks_df['CCS'] = peaks_df['CCS'].astype(np.float32)
            
            # Recherche vectorisée des correspondances
            matches_df = self.find_matches_vectorized(
                peaks_df=peaks_df,
                tolerances=self.config.tolerances
            )

            if matches_df.empty:
                self.logger.warning("Aucune correspondance trouvée.")
                return None

            # Sauvegarde des résultats
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            matches_path = output_path / 'all_matches.parquet'
            matches_df.to_parquet(matches_path)

            self.logger.info(self._get_identification_stats(matches_df))
            
            return matches_df

        except Exception as e:
            self.logger.error(f"Erreur lors de l'identification des composés : {str(e)}")
            raise

    def _get_identification_stats(self, matches_df: pd.DataFrame) -> str:
        """
        Génère un résumé des statistiques d'identification.
        
        Args:
            matches_df: DataFrame des correspondances
            
        Returns:
            str: Résumé des statistiques
        """
        stats = []
        total_matches = len(matches_df)
        unique_compounds = matches_df['match_name'].nunique()
        stats.append(f"Total des correspondances : {total_matches}")
        stats.append(f"Composés uniques : {unique_compounds}")
        
        if 'confidence_level' in matches_df.columns:
            for level in sorted(matches_df['confidence_level'].unique()):
                level_count = len(matches_df[matches_df['confidence_level'] == level])
                level_percent = (level_count / total_matches) * 100
                stats.append(f"Niveau {level}: {level_count} ({level_percent:.1f}%)")
        
        return "\n".join(stats)

    def get_identification_metrics(self, matches_df: pd.DataFrame) -> Dict:
        """
        Calcule les métriques d'identification.
        
        Args:
            matches_df: DataFrame des correspondances
            
        Returns:
            Dict: Métriques calculées
        """
        try:
            metrics = {
                'total_matches': len(matches_df),
                'unique_compounds': matches_df['match_name'].nunique(),
                'confidence_levels': {},
                'mass_error_stats': {},
                'rt_error_stats': {},
                'ccs_error_stats': {}
            }

            # Statistiques par niveau de confiance
            if 'confidence_level' in matches_df.columns:
                for level in sorted(matches_df['confidence_level'].unique()):
                    level_df = matches_df[matches_df['confidence_level'] == level]
                    metrics['confidence_levels'][f'level_{level}'] = {
                        'count': len(level_df),
                        'percent': (len(level_df) / len(matches_df)) * 100,
                        'unique_compounds': level_df['match_name'].nunique()
                    }

            # Statistiques d'erreurs
            if 'mz_error_ppm' in matches_df.columns:
                metrics['mass_error_stats'] = self._calculate_error_stats(
                    matches_df['mz_error_ppm']
                )
            
            if 'rt_error_min' in matches_df.columns:
                metrics['rt_error_stats'] = self._calculate_error_stats(
                    matches_df['rt_error_min']
                )
            
            if 'ccs_error_percent' in matches_df.columns:
                metrics['ccs_error_stats'] = self._calculate_error_stats(
                    matches_df['ccs_error_percent']
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques : {str(e)}")
            return {}

    def _calculate_error_stats(self, error_series: pd.Series) -> Dict:
        """
        Calcule les statistiques d'erreur.
        
        Args:
            error_series: Série des erreurs
            
        Returns:
            Dict: Statistiques calculées
        """
        return {
            'mean': float(error_series.mean()),
            'std': float(error_series.std()),
            'median': float(error_series.median()),
            'min': float(error_series.min()),
            'max': float(error_series.max()),
            'abs_mean': float(error_series.abs().mean())
        }
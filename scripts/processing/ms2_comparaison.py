import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
from ..utils.matching_utils import assign_confidence_level

class MS2Comparator:
    def __init__(self, tolerance_mz: float = 0.01):
        self.tolerance_mz = tolerance_mz
        self.logger = logging.getLogger(__name__)
        self.n_workers = max(1, int(mp.cpu_count() * 0.75))

    def normalize_spectrum(self, mz: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalisation vectorisée des spectres."""
        if len(mz) == 0 or len(intensity) == 0:
            return np.array([]), np.array([])
            
        # Filtrer les intensités nulles
        valid_mask = intensity > 0
        if not valid_mask.any():
            return np.array([]), np.array([])
            
        mz_filtered = mz[valid_mask]
        intensity_filtered = intensity[valid_mask]
        
        max_intensity = np.max(intensity_filtered)
        if max_intensity == 0:
            return np.array([]), np.array([])
            
        return mz_filtered, (intensity_filtered / max_intensity) * 1000

    def calculate_cosine_similarity(self, spec1_mz: np.ndarray, spec1_int: np.ndarray, 
                                  spec2_mz: np.ndarray, spec2_int: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux spectres."""
        if len(spec1_mz) == 0 or len(spec2_mz) == 0:
            return 0.0
            
        # Créer des vecteurs alignés
        all_mz = np.union1d(spec1_mz, spec2_mz)
        vec1 = np.zeros(len(all_mz))
        vec2 = np.zeros(len(all_mz))
        
        # Remplir les vecteurs avec correspondances
        for i, mz in enumerate(all_mz):
            # Trouver les pics correspondants dans le spectre 1
            matches1 = np.abs(spec1_mz - mz) <= self.tolerance_mz
            if matches1.any():
                vec1[i] = np.max(spec1_int[matches1])
                
            # Trouver les pics correspondants dans le spectre 2
            matches2 = np.abs(spec2_mz - mz) <= self.tolerance_mz
            if matches2.any():
                vec2[i] = np.max(spec2_int[matches2])
        
        # Calculer la similarité cosinus
        if np.sum(vec1) == 0 or np.sum(vec2) == 0:
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def calculate_similarity_batch(self, batch_data: Tuple[pd.Series, Dict]) -> Tuple[int, float]:
        """Calcule la similarité pour un lot de données."""
        row, ref_spectra_dict = batch_data
        
        try:
            best_score = 0.0
            exp_mz = np.array(row['peaks_mz_ms2'])
            exp_int = np.array(row['peaks_intensities_ms2'])
            
            if len(exp_mz) == 0 or len(exp_int) == 0:
                return row.name, 0.0

            # Normaliser le spectre expérimental
            exp_mz_norm, exp_int_norm = self.normalize_spectrum(exp_mz, exp_int)
            
            if len(exp_mz_norm) == 0:
                return row.name, 0.0
            
            # Rechercher les spectres de référence
            molecule_name = row['match_name']
            
            # Essayer différentes clés de recherche
            search_keys = [
                molecule_name,  # Nom seul
                f"{molecule_name}_{row['match_adduct']}"  # Nom + adduit
            ]
            
            for key in search_keys:
                if key in ref_spectra_dict:
                    for ref_mz, ref_int, energy in ref_spectra_dict[key]:
                        if len(ref_mz) > 0 and len(ref_int) > 0:
                            similarity = self.calculate_cosine_similarity(
                                exp_mz_norm, exp_int_norm, ref_mz, ref_int
                            )
                            if similarity > best_score:
                                best_score = similarity
                                # Log pour déboguer
                                if similarity > 0.2:
                                    self.logger.debug(f"Meilleur score {similarity:.3f} pour {molecule_name} à {energy} eV")

            return row.name, best_score
            
        except Exception as e:
            self.logger.error(f"Erreur similarité pour {row.get('match_name', 'unknown')}: {str(e)}")
            return row.name, 0.0

def is_valid_ms2_data(mz_data, int_data):
    """Vérifie si les données MS2 sont valides."""
    try:
        # Vérifier les valeurs scalaires NaN
        if isinstance(mz_data, (int, float)) and pd.isna(mz_data):
            return False
        if isinstance(int_data, (int, float)) and pd.isna(int_data):
            return False
        
        # Vérifier les listes
        if isinstance(mz_data, list) and isinstance(int_data, list):
            return len(mz_data) > 0 and len(int_data) > 0 and len(mz_data) == len(int_data)
        
        # Vérifier arrays numpy
        if isinstance(mz_data, np.ndarray) and isinstance(int_data, np.ndarray):
            return mz_data.size > 0 and int_data.size > 0 and mz_data.size == int_data.size
        
        return False
    except:
        return False

def add_ms2_scores(matches_df: pd.DataFrame, identifier: object) -> None:
    """
    Ajoute les scores de similarité MS2 en utilisant toutes les énergies de collision.
    """
    try:
        print("\n🔬 Analyse des spectres MS2 (toutes énergies)...")
        comparator = MS2Comparator(tolerance_mz=0.01)
        
        # Initialisation
        matches_df['ms2_similarity_score'] = 0.0
        
        # Pré-filtrage
        matches_to_analyze = matches_df[
            (matches_df['has_ms2_db'] == 1) &
            matches_df['peaks_mz_ms2'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        
        n_matches_with_ms2 = len(matches_to_analyze)
        print(f"   ✓ {n_matches_with_ms2}/{len(matches_df)} matches avec MS2 à analyser")
        
        if n_matches_with_ms2 == 0:
            print("   ℹ️ Aucune correspondance avec MS2 à analyser")
            return
        
        # Préparation du cache avec toutes les énergies
        ref_spectra_dict = {}
        
        # Filtrer pour avoir seulement les spectres MS2 valides
        db_with_ms2 = identifier.db[identifier.db['has_ms2_db'] == 1]
        print(f"   ✓ {len(db_with_ms2)} spectres MS2 valides dans la base")
        
        # Extraire les spectres par molécule pour toutes les énergies
        molecules_processed = set()
        for name in matches_to_analyze['match_name'].unique():
            if name in molecules_processed:
                continue
                
            # Rechercher les spectres pour cette molécule
            ref_spectra = db_with_ms2[db_with_ms2['Name'] == name]
            
            if not ref_spectra.empty:
                normalized_spectra = []
                energy_count = {}
                
                for _, ref_row in ref_spectra.iterrows():
                    try:
                        # Vérifier que les données MS2 sont valides
                        mz_data = ref_row['peaks_ms2_mz']
                        int_data = ref_row['peaks_ms2_intensities']
                        
                        if not is_valid_ms2_data(mz_data, int_data):
                            continue
                        
                        # Convertir en arrays numpy
                        ref_mz = np.array(mz_data)
                        ref_int = np.array(int_data)
                        
                        # Vérifier que les arrays sont valides
                        if len(ref_mz) > 0 and len(ref_int) > 0 and len(ref_mz) == len(ref_int):
                            ref_mz_norm, ref_int_norm = comparator.normalize_spectrum(ref_mz, ref_int)
                            
                            if len(ref_mz_norm) > 0:
                                # Récupérer l'énergie de collision
                                energy = ref_row.get('Collision_energy_clean', 'unknown')
                                if pd.isna(energy):
                                    energy = ref_row.get('Collision_energy', 'unknown')
                                
                                normalized_spectra.append((ref_mz_norm, ref_int_norm, energy))
                                energy_count[energy] = energy_count.get(energy, 0) + 1
                                
                    except Exception as e:
                        comparator.logger.debug(f"Erreur conversion spectre pour {name}: {e}")
                        continue
                
                if normalized_spectra:
                    # Stocker avec différentes clés pour la recherche
                    ref_spectra_dict[name] = normalized_spectra
                    
                    # Ajouter les combinaisons avec adduits
                    for adduct in ref_spectra['adduct'].unique():
                        if pd.notna(adduct):
                            key = f"{name}_{adduct}"
                            ref_spectra_dict[key] = normalized_spectra
                    
                    # Afficher les énergies disponibles
                    energy_str = ', '.join([f"{e}: {c}" for e, c in energy_count.items()])
                    print(f"   ✓ {len(normalized_spectra)} spectres chargés pour {name} ({energy_str})")
            
            molecules_processed.add(name)
        
        print(f"   ✓ {len(ref_spectra_dict)} entrées de spectres de référence chargées")
        
        if len(ref_spectra_dict) == 0:
            print("   ❌ Aucun spectre de référence valide trouvé")
            return
        
        # Traitement parallèle
        batch_data = [(row, ref_spectra_dict) for _, row in matches_to_analyze.iterrows()]
        
        with ProcessPoolExecutor(max_workers=comparator.n_workers) as executor:
            futures = [executor.submit(comparator.calculate_similarity_batch, data) 
                      for data in batch_data]
            
            for future in tqdm(futures, total=len(batch_data), desc="Calcul scores MS2"):
                idx, score = future.result()
                matches_df.loc[idx, 'ms2_similarity_score'] = score

        # Recalcul des niveaux de confiance
        print("\n📊 Recalcul des niveaux de confiance...")
        for idx in tqdm(matches_df.index, desc="Attribution niveaux"):
            confidence_level, confidence_reason = assign_confidence_level(matches_df.loc[idx])
            matches_df.loc[idx, ['confidence_level', 'confidence_reason']] = [confidence_level, confidence_reason]

        # Statistiques finales
        ms2_scores = matches_df['ms2_similarity_score']
        high_scores = (ms2_scores >= 0.7).sum()
        medium_scores = ((ms2_scores >= 0.4) & (ms2_scores < 0.7)).sum()
        low_scores = ((ms2_scores >= 0.2) & (ms2_scores < 0.4)).sum()
        zero_scores = (ms2_scores == 0.0).sum()
        
        print(f"\n📈 Statistiques des scores MS2:")
        print(f"   • Scores élevés (≥0.7): {high_scores}")
        print(f"   • Scores moyens (0.4-0.7): {medium_scores}")
        print(f"   • Scores faibles (0.2-0.4): {low_scores}")
        print(f"   • Pas de similarité (0.0): {zero_scores}")
        
        # Exemples de molécules avec scores élevés
        high_score_molecules = matches_df[matches_df['ms2_similarity_score'] >= 0.2]
        if len(high_score_molecules) > 0:
            print(f"   • Molécules avec scores MS2 détectables: {len(high_score_molecules)} molécules")
            
            # Afficher les 5 meilleurs scores
            best_matches = high_score_molecules.nlargest(5, 'ms2_similarity_score')
            print("   • Top 5 des meilleurs scores:")
            for _, match in best_matches.iterrows():
                print(f"      - {match['match_name']}: {match['ms2_similarity_score']:.3f}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Erreur lors du calcul des scores MS2: {str(e)}")
        raise
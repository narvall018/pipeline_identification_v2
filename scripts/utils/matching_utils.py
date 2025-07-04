#scripts/utils/matching_utils.py
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from ..config.config import Config
from typing import Optional, Dict, Tuple, Any

def calculate_match_scores(
    match: Dict[str, float],
    tolerances: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calcule les scores individuels et globaux pour une correspondance.
    """
    if tolerances is None:
        tolerances = {
            'mz_ppm': 10,
            'ccs_percent': 12,
            'rt_min': 2
        }

    weights = {'mz': 0.4, 'ccs': 0.4, 'rt': 0.2}
    scores = {}

    # Score m/z
    scores['mz'] = max(0, 1 - abs(match['mz_error_ppm']) / tolerances['mz_ppm'])

    # Score CCS
    if pd.notna(match['match_ccs_exp']):
        ccs_error = abs(match['ccs_error_percent'])
        weights['ccs'] *= 1.2
        scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
    elif pd.notna(match['match_ccs_pred']):
        match['ccs_error_percent'] = (match['peak_ccs'] - match['match_ccs_pred']) / match['match_ccs_pred'] * 100
        ccs_error = abs(match['ccs_error_percent'])
        weights['ccs'] *= 0.6
        scores['ccs'] = max(0, 1 - ccs_error / tolerances['ccs_percent'])
    else:
        scores['ccs'] = 0

    # Score RT
    if pd.notna(match['match_rt_obs']):
        rt_error = abs(match['rt_error_min'])
        weights['rt'] *= 1.2
        scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
    elif pd.notna(match['match_rt_pred']):
        match['rt_error_min'] = abs(match['peak_rt'] - match['match_rt_pred'])
        rt_error = match['rt_error_min']
        weights['rt'] *= 0.6
        scores['rt'] = max(0, 1 - rt_error / tolerances['rt_min'])
    else:
        scores['rt'] = 0

    # Normalisation des poids
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Score global
    global_score = sum(scores[key] * weights[key] for key in weights)

    return {
        'individual_scores': scores,
        'global_score': global_score,
        'ccs_source': 'exp' if pd.notna(match['match_ccs_exp']) else 'pred',
        'rt_source': 'obs' if pd.notna(match['match_rt_obs']) else 'pred'
    }

def assign_confidence_level_schymanski(
    match: Dict[str, Any], 
    tolerances: Optional[Dict[str, float]] = None
) -> Tuple[int, str]:
    """
    Assigne un niveau de confiance selon les critères de Schymanski et al.
    
    Niveau 1: Masse exacte + RT + MS/MS + CCS + Standard de référence
    Niveau 2: Masse exacte + MS/MS avec base de données + CCS avec références
    Niveau 3: Masse exacte + MS/MS partiel + CCS avec prédictions
    Niveau 4: Masse exacte + CCS aide à la distinction
    Niveau 5: Masse exacte seulement
    """
    if tolerances is None:
        tolerances = {
            'mz_ppm': 10,
            'ccs_percent': 12,
            'rt_min': 2.0,
            'ms2_score_high': 0.7,  # Score élevé pour correspondance parfaite
            'ms2_score_medium': 0.4,  # Score moyen pour correspondance partielle
            'ms2_score_low': 0.2   # Score minimum pour correspondance détectable
        }

    # Vérification m/z (requis pour tous les niveaux)
    mz_match = abs(match['mz_error_ppm']) <= tolerances['mz_ppm']
    if not mz_match:
        return 5, "Erreur m/z > tolérance"

    # Détection des données disponibles
    has_rt_reference = pd.notna(match['match_rt_obs'])  # RT observé = standard de référence
    has_rt_prediction = pd.notna(match['match_rt_pred'])  # RT prédit
    has_ccs_reference = pd.notna(match['match_ccs_exp'])  # CCS expérimental = standard de référence
    has_ccs_prediction = pd.notna(match['match_ccs_pred'])  # CCS prédit
    has_ms2_db = match.get('has_ms2_db', 0) == 1
    has_ms2_peaks = isinstance(match.get('peaks_intensities_ms2', []), (list, np.ndarray)) and len(match.get('peaks_intensities_ms2', [])) > 0
    
    # Vérification des correspondances
    rt_reference_match = has_rt_reference and abs(match['rt_error_min']) <= tolerances['rt_min']
    rt_prediction_match = has_rt_prediction and abs(match['rt_error_min']) <= tolerances['rt_min']
    ccs_reference_match = has_ccs_reference and abs(match['ccs_error_percent']) <= tolerances['ccs_percent']
    ccs_prediction_match = has_ccs_prediction and abs(match['ccs_error_percent']) <= tolerances['ccs_percent']
    
    # Vérification MS2
    ms2_score = match.get('ms2_similarity_score', 0)
    ms2_perfect_match = ms2_score >= tolerances['ms2_score_high']
    ms2_good_match = ms2_score >= tolerances['ms2_score_medium']
    ms2_partial_match = ms2_score >= tolerances['ms2_score_low']

    # **NIVEAU 1** : Masse exacte + RT + MS/MS + CCS + Standard de référence
    # Tous les critères doivent être remplis avec des standards de référence
    if (rt_reference_match and 
        ccs_reference_match and 
        has_ms2_db and has_ms2_peaks and ms2_perfect_match):
        return 1, "Match parfait avec standards de référence (m/z + RT obs + CCS exp + MS/MS parfait)"

    # **NIVEAU 2** : Masse exacte + MS/MS avec base de données + CCS avec références
    # Pas de RT requis, mais MS/MS et CCS doivent être bons
    if (ccs_reference_match and 
        has_ms2_db and has_ms2_peaks and ms2_good_match):
        return 2, "Match probable avec références (m/z + CCS exp + MS/MS base de données)"

    # **NIVEAU 3** : Masse exacte + MS/MS partiel + CCS avec prédictions
    # MS/MS partiel acceptable, CCS peut être prédit
    if ((ccs_reference_match or ccs_prediction_match) and 
        has_ms2_db and has_ms2_peaks and ms2_partial_match):
        return 3, "Match possible avec prédictions (m/z + CCS + MS/MS partiel)"

    # **NIVEAU 4** : Masse exacte + CCS aide à la distinction
    # CCS aide à distinguer, pas de MS/MS interprétable requis
    if ccs_reference_match or ccs_prediction_match:
        return 4, "Match tentatif avec aide CCS (m/z + CCS pour distinction)"

    # **NIVEAU 5** : Masse exacte seulement
    # Seule la masse exacte correspond
    return 5, "Match incertain (m/z uniquement)"

def assign_confidence_level(
    match: Dict[str, Any], 
    tolerances: Optional[Dict[str, float]] = None
) -> Tuple[int, str]:
    """
    Wrapper pour la nouvelle fonction de niveaux de confiance Schymanski.
    """
    return assign_confidence_level_schymanski(match, tolerances)

def find_matches_asof(
    peaks_df: pd.DataFrame,
    db_df: pd.DataFrame,
    tolerances: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Trouve des correspondances entre les pics et la base de données en utilisant `merge_asof`.
    """
    if tolerances is None:
        tolerances = Config.IDENTIFICATION.tolerances

    peaks_df = peaks_df.sort_values('mz')
    db_df = db_df.sort_values('mz')

    mz_tolerance = peaks_df['mz'].mean() * tolerances['mz_ppm'] * 1e-6

    matches = pd.merge_asof(
        peaks_df,
        db_df,
        on='mz',
        direction='nearest',
        tolerance=mz_tolerance
    )

    return matches

def find_matches_window(
    peaks_df: pd.DataFrame,
    db_df: pd.DataFrame,
    tolerances: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Trouve les correspondances entre les pics et une base de données de molécules.
    """
    if tolerances is None:
        tolerances = {
            'mz_ppm': 10,
            'ccs_percent': 12,
            'rt_min': 2
        }

    # Préparation de la base de données
    db_df['Name_str'] = db_df['Name'].astype(str).fillna('')
    db_df['adduct_str'] = db_df['adduct'].astype(str).fillna('')
    db_df['SMILES_str'] = db_df['SMILES'].astype(str).fillna('')

    db_df['molecule_id'] = db_df.apply(
        lambda row: f"{row['Name_str']}_{row['adduct_str']}" if row['Name_str'] and row['adduct_str'] else row['SMILES_str'],
        axis=1
    )

    # NOUVELLE VALIDATION MS2 COHÉRENTE
    def validate_ms2_for_matching(row):
        """Validation MS2 robuste pour le matching"""
        mz_data = row['peaks_ms2_mz']
        int_data = row['peaks_ms2_intensities']
        
        # Vérifier que les deux champs existent et sont des listes non vides
        mz_valid = isinstance(mz_data, (list, np.ndarray)) and len(mz_data) > 0
        int_valid = isinstance(int_data, (list, np.ndarray)) and len(int_data) > 0
        
        # Vérifier que les longueurs correspondent
        if mz_valid and int_valid:
            return len(mz_data) == len(int_data)
        
        return False

    # Si has_ms2_db existe déjà (pré-traitement), on le préserve et vérifie
    if 'has_ms2_db' not in db_df.columns:
        db_df['has_ms2_db'] = db_df.apply(validate_ms2_for_matching, axis=1).astype(int)
    else:
        # Vérifier la cohérence et corriger si nécessaire
        db_df['has_ms2_db_corrected'] = db_df.apply(validate_ms2_for_matching, axis=1).astype(int)
        
        # Comparer et corriger les incohérences
        inconsistent_mask = db_df['has_ms2_db'] != db_df['has_ms2_db_corrected']
        if inconsistent_mask.any():
            db_df['has_ms2_db'] = db_df['has_ms2_db_corrected']
        
        # Nettoyer la colonne temporaire
        db_df = db_df.drop(columns=['has_ms2_db_corrected'])

    # Agrégation par molécule SANS drop/merge pour éviter les pertes
    molecule_ms2_status = db_df.groupby('molecule_id')['has_ms2_db'].max()

    # Application directe sans merge pour préserver l'intégrité
    for molecule_id, max_status in molecule_ms2_status.items():
        mask = db_df['molecule_id'] == molecule_id
        db_df.loc[mask, 'has_ms2_db'] = max_status

    # Tri pour recherche optimisée
    db_df = db_df.sort_values('mz').reset_index(drop=True)
    db_mz = db_df['mz'].values

    all_matches = []

    # Recherche des correspondances
    for peak in peaks_df.itertuples():
        mz_tolerance = peak.mz * tolerances['mz_ppm'] * 1e-6
        mz_min, mz_max = peak.mz - mz_tolerance, peak.mz + mz_tolerance

        idx_start = np.searchsorted(db_mz, mz_min, side='left')
        idx_end = np.searchsorted(db_mz, mz_max, side='right')
        matches = db_df.iloc[idx_start:idx_end]

        if not matches.empty:
            for match in matches.itertuples():
                # Vérification RT
                rt_error, rt_match = None, False
                if pd.notna(match.Observed_RT):
                    rt_error = abs(peak.retention_time - match.Observed_RT)
                    rt_match = rt_error <= tolerances['rt_min']
                elif pd.notna(match.Predicted_RT):
                    rt_error = abs(peak.retention_time - match.Predicted_RT)
                    rt_match = rt_error <= tolerances['rt_min']

                # Vérification CCS
                ccs_error, ccs_match = None, False
                if pd.notna(match.ccs_exp):
                    ccs_error = abs((peak.CCS - match.ccs_exp) / match.ccs_exp * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']
                elif pd.notna(match.ccs_pred):
                    ccs_error = abs((peak.CCS - match.ccs_pred) / match.ccs_pred * 100)
                    ccs_match = ccs_error <= tolerances['ccs_percent']

                # Ajout de la correspondance si elle respecte les tolérances
                if (rt_match or rt_error is None) and (ccs_match or ccs_error is None):
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
                        'rt_error_min': rt_error,
                        'ccs_error_percent': ccs_error,
                        'has_ms2_db': match.has_ms2_db,
                        'molecule_id': match.molecule_id,
                        'daphnia_LC50_48_hr_ug/L': float(matches.iloc[match.Index]['LC50_48_hr_ug/L']) if pd.notna(matches.iloc[match.Index]['LC50_48_hr_ug/L']) else None,
                        'algae_EC50_72_hr_ug/L': float(matches.iloc[match.Index]['EC50_72_hr_ug/L']) if pd.notna(matches.iloc[match.Index]['EC50_72_hr_ug/L']) else None,
                        'pimephales_LC50_96_hr_ug/L': float(matches.iloc[match.Index]['LC50_96_hr_ug/L']) if pd.notna(matches.iloc[match.Index]['LC50_96_hr_ug/L']) else None
                    }

                    # Calcul des scores et niveau de confiance
                    score_details = calculate_match_scores(match_details)
                    confidence_level, confidence_reason = assign_confidence_level(match_details)

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
        # Tri pour prioriser les correspondances avec MS2 et le score global
        matches_df = matches_df.sort_values(
            ['molecule_id', 'has_ms2_db', 'global_score'],
            ascending=[True, False, False]
        )
        # Suppression des doublons
        matches_df = matches_df.drop_duplicates(subset='molecule_id', keep='first')

        # Tri final par niveau de confiance et score global
        matches_df = matches_df.sort_values(['confidence_level', 'global_score'], ascending=[True, False])

    return matches_df
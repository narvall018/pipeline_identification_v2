#scripts/processing/feature_matrix.py
# -*- coding:utf-8 -*-

import logging
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
from ..processing.identification import CompoundIdentifier
from ..processing.ms2_comparaison import add_ms2_scores
from ..utils.matching_utils import find_matches_window
from ..config.config import Config

class FeatureProcessor:
    """Classe pour le traitement et l'alignement des features."""
    
    def __init__(self):
        """Initialise le processeur de features avec la configuration."""
        self.config = Config.FEATURE_ALIGNMENT
        self.logger = logging.getLogger(__name__)

    def align_features_across_samples(
        self,
        samples_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Aligne les features entre plusieurs échantillons.
        
        Args:
            samples_dir: Répertoire contenant les sous-dossiers d'échantillons
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: 
                (intensity_matrix, feature_df, raw_files)
        """
        print("\n🔄 Alignement des features entre échantillons...")
        
        all_peaks = []
        sample_names = []
        
        # Chargement des pics pour chaque échantillon
        for sample_dir in samples_dir.glob("*"):
            if sample_dir.is_dir():
                peaks_file = sample_dir / "ms1" / "common_peaks.parquet"
                if peaks_file.exists():
                    peaks = pd.read_parquet(peaks_file)
                    if not peaks.empty:
                        print(f"   ✓ Chargement de {sample_dir.name}: {len(peaks)} pics")
                        peaks = peaks.assign(
                            sample=sample_dir.name,
                            orig_rt=peaks['retention_time'],
                            orig_dt=peaks['drift_time']
                        )
                        all_peaks.append(peaks)
                        sample_names.append(sample_dir.name)
        
        if not all_peaks:
            raise ValueError("Aucun pic trouvé dans les échantillons")
        
        # Fusion des données de tous les échantillons
        df = pd.concat(all_peaks, ignore_index=True)
        print(f"   ✓ Total: {len(df)} pics à travers {len(sample_names)} échantillons")
        
        print("\n🎯 Clustering des features...")
        X = df[['mz', 'drift_time', 'retention_time']].to_numpy()
        median_mz = np.median(X[:, 0])
        
        X_scaled = np.column_stack([
            X[:, 0] / (median_mz * self.config.mz_ppm * 1e-6),
            X[:, 1] / self.config.dt_tolerance,
            X[:, 2] / self.config.rt_tolerance
        ])
        
        # Clustering avec DBSCAN
        clusters = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            algorithm=self.config.algorithm,
            n_jobs=-1
        ).fit_predict(X_scaled)
        
        df['cluster'] = clusters
        non_noise_clusters = np.unique(clusters[clusters != -1])
        
        print("\n📊 Génération des features alignées...")
        cluster_groups = df[df['cluster'].isin(non_noise_clusters)].groupby('cluster')
        features = []
        intensities = {}
        
        for cluster_id, cluster_data in cluster_groups:
            max_intensity_idx = cluster_data['intensity'].idxmax()
            max_intensity_row = cluster_data.loc[max_intensity_idx]
            
            feature = {
                'mz': cluster_data['mz'].mean(),
                'retention_time': cluster_data['retention_time'].mean(),
                'drift_time': cluster_data['drift_time'].mean(),
                'intensity': max_intensity_row['intensity'],
                'source_sample': max_intensity_row['sample'],
                'source_rt': max_intensity_row['orig_rt'],
                'source_dt': max_intensity_row['orig_dt'],
                'n_samples': cluster_data['sample'].nunique(),
                'samples': ','.join(sorted(cluster_data['sample'].unique())),
                'feature_id': f"F{len(features) + 1:04d}"
            }
            
            if 'CCS' in cluster_data.columns:
                feature['CCS'] = cluster_data['CCS'].mean()
            
            features.append(feature)
            
            feature_name = f"{feature['feature_id']}_mz{feature['mz']:.4f}"
            sample_intensities = cluster_data.groupby('sample')['intensity'].max()
            intensities[feature_name] = sample_intensities
        
        feature_df = pd.DataFrame(features)
        print(f"   ✓ {len(feature_df)} features uniques détectées")
        
        # Création de la matrice d'intensités
        intensity_matrix = pd.DataFrame(intensities, index=sample_names).fillna(0)
        
        # Mapping des fichiers raw
        raw_files = {
            sample_dir.name: next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"))
            for sample_dir in samples_dir.glob("*")
            if sample_dir.is_dir() and next(Path("data/input/samples").glob(f"{sample_dir.name}*.parquet"), None)
        }
        
        return intensity_matrix, feature_df, raw_files

    def process_features(
        self,
        feature_df: pd.DataFrame,
        raw_files: Dict,
        identifier: CompoundIdentifier
    ) -> pd.DataFrame:
        """
        Traite les features pour l'identification et l'extraction MS2.
        
        Args:
            feature_df: DataFrame des features
            raw_files: Dictionnaire des fichiers raw
            identifier: Instance de CompoundIdentifier
            
        Returns:
            pd.DataFrame: Features identifiées avec MS2
        """
        try:
            # Chargement des données MS2
            ms2_data = self._load_ms2_data(raw_files)
            
            # Extraction des spectres MS2 EXPÉRIMENTAUX (vos données)
            feature_df = self._extract_ms2_spectra_optimized(feature_df, ms2_data)
            n_with_ms2 = sum(1 for x in feature_df['peaks_mz_ms2'] if len(x) > 0)
            print(f"{n_with_ms2}/{len(feature_df)} features avec spectres MS2 expérimentaux")

            # Identification des composés
            matches = self._identify_features(feature_df, identifier)
            
            if not matches.empty:
                # Ajout des scores MS2 et ajout des spectres de référence
                matches = self._add_ms2_scores_and_add_reference_spectra(matches, identifier)
                
                # Tri final des colonnes
                cols = [col for col in matches.columns if col != 'confidence_level'] + ['confidence_level']
                matches = matches[cols]
                
            return matches

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des features : {str(e)}")
            raise

    def _load_ms2_data(self, raw_files: Dict) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données MS2 en une fois."""
        ms2_data = {}
        for sample_name, raw_file in raw_files.items():
            raw_data = pd.read_parquet(raw_file)
            ms2_data[sample_name] = raw_data[raw_data['mslevel'].astype(int) == 2].copy()
        return ms2_data

    def _extract_ms2_spectra_optimized(
        self,
        feature_df: pd.DataFrame,
        ms2_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Extrait les spectres MS2 de manière optimisée par échantillon."""
        # Initialisation des colonnes MS2
        feature_df['peaks_mz_ms2'] = [[] for _ in range(len(feature_df))]
        feature_df['peaks_intensities_ms2'] = [[] for _ in range(len(feature_df))]

        # Grouper les features par échantillon source
        grouped_features = feature_df.groupby('source_sample')

        for sample_name, group in tqdm(grouped_features, desc="Extraction MS2"):
            if sample_name not in ms2_data:
                continue

            sample_ms2 = ms2_data[sample_name]
            
            for idx, feature in group.iterrows():
                rt_min = feature['source_rt'] - Config.MS2_EXTRACTION.rt_tolerance
                rt_max = feature['source_rt'] + Config.MS2_EXTRACTION.rt_tolerance
                dt_min = feature['source_dt'] - Config.MS2_EXTRACTION.dt_tolerance
                dt_max = feature['source_dt'] + Config.MS2_EXTRACTION.dt_tolerance

                # Extraire les spectres MS2 correspondants
                ms2_window = sample_ms2[
                    (sample_ms2['rt'].between(rt_min, rt_max)) &
                    (sample_ms2['dt'].between(dt_min, dt_max))
                ]

                if not ms2_window.empty:
                    # Traitement du spectre MS2
                    ms2_window['mz_rounded'] = ms2_window['mz'].round(
                        Config.MS2_EXTRACTION.mz_round_decimals
                    )
                    spectrum = ms2_window.groupby('mz_rounded')['intensity'].sum().reset_index()

                    max_intensity = spectrum['intensity'].max()
                    if max_intensity > 0:
                        spectrum['intensity_normalized'] = (
                            spectrum['intensity'] / max_intensity * 
                            Config.MS2_EXTRACTION.intensity_scale
                        ).round(0).astype(int)
                        
                        spectrum = spectrum.nlargest(
                            Config.MS2_EXTRACTION.max_peaks,
                            'intensity'
                        )
                        
                        feature_df.at[idx, 'peaks_mz_ms2'] = spectrum['mz_rounded'].tolist()
                        feature_df.at[idx, 'peaks_intensities_ms2'] = spectrum['intensity_normalized'].tolist()

        return feature_df

    def _identify_features(
        self,
        feature_df: pd.DataFrame,
        identifier: CompoundIdentifier
    ) -> pd.DataFrame:
        """Identifie les features dans la base de données."""
        db = identifier.db.copy()
        db = db.sort_values('mz')
        db_mz = db['mz'].values

        all_matches = []
        
        for idx, feature in tqdm(feature_df.iterrows(), total=len(feature_df),
                               desc="Identification"):
            mz_tolerance = feature['mz'] * Config.IDENTIFICATION.tolerances['mz_ppm'] * 1e-6
            mz_min, mz_max = feature['mz'] - mz_tolerance, feature['mz'] + mz_tolerance

            idx_start = np.searchsorted(db_mz, mz_min, side='left')
            idx_end = np.searchsorted(db_mz, mz_max, side='right')

            if idx_start == idx_end:
                continue

            feature_data = pd.DataFrame([{
                'mz': feature['mz'],
                'retention_time': feature['retention_time'],
                'drift_time': feature['drift_time'],
                'CCS': feature['CCS'],
                'intensity': feature['intensity']
            }])

            matches_for_feature = find_matches_window(
                feature_data,
                db.iloc[idx_start:idx_end]
            )
            
            if not matches_for_feature.empty:
                matches_for_feature['feature_idx'] = idx
                matches_for_feature['peaks_mz_ms2'] = [feature['peaks_mz_ms2']] * len(matches_for_feature)
                matches_for_feature['peaks_intensities_ms2'] = [feature['peaks_intensities_ms2']] * len(matches_for_feature)
                all_matches.append(matches_for_feature)

        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        return pd.DataFrame()

    def _add_ms2_scores_and_add_reference_spectra(
        self,
        matches_df: pd.DataFrame,
        identifier: CompoundIdentifier
    ) -> pd.DataFrame:
        """
        Ajoute les scores MS2 et ajoute les spectres de référence en sélectionnant 
        le spectre avec la meilleure similarité.
        """
        # Calculer les scores MS2 en utilisant vos spectres expérimentaux
        # Temporairement capturer la sortie pour la supprimer
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            add_ms2_scores(matches_df, identifier)
        finally:
            sys.stdout = old_stdout
        
        # Importer la classe MS2Comparator pour calculer les similarités
        from ..processing.ms2_comparaison import MS2Comparator
        comparator = MS2Comparator(tolerance_mz=0.01)
        
        # Préparer un dictionnaire des meilleurs spectres par molécule
        best_spectra_dict = {}
        
        # Filtrer la base pour avoir seulement les spectres MS2 valides
        db_with_ms2 = identifier.db[identifier.db['has_ms2_db'] == 1]
        
        # Extraire les spectres par molécule en testant toutes les énergies
        molecules_processed = set()
        molecules_optimized = 0
        
        for name in matches_df['match_name'].unique():
            if name in molecules_processed or pd.isna(name):
                continue
                
            # Rechercher les spectres pour cette molécule
            molecule_spectra = db_with_ms2[db_with_ms2['Name'] == name]
            
            if not molecule_spectra.empty:
                # Récupérer le spectre expérimental correspondant
                molecule_matches = matches_df[matches_df['match_name'] == name]
                if molecule_matches.empty:
                    continue
                    
                # Prendre le premier match pour avoir le spectre expérimental
                exp_match = molecule_matches.iloc[0]
                exp_mz = exp_match.get('peaks_mz_ms2', [])
                exp_int = exp_match.get('peaks_intensities_ms2', [])
                
                # Vérifier que le spectre expérimental est valide
                if not (isinstance(exp_mz, list) and len(exp_mz) > 0 and
                        isinstance(exp_int, list) and len(exp_int) > 0):
                    continue
                
                # Normaliser le spectre expérimental
                exp_mz_norm, exp_int_norm = comparator.normalize_spectrum(
                    np.array(exp_mz), np.array(exp_int)
                )
                
                if len(exp_mz_norm) == 0:
                    continue
                
                # Tester tous les spectres de référence disponibles
                best_similarity = 0.0
                best_spectrum = None
                best_energy = "inconnue"
                
                for _, ref_row in molecule_spectra.iterrows():
                    try:
                        # Vérifier que les données MS2 sont valides
                        mz_data = ref_row['peaks_ms2_mz']
                        int_data = ref_row['peaks_ms2_intensities']
                        
                        if not (isinstance(mz_data, list) and len(mz_data) > 0 and
                                isinstance(int_data, list) and len(int_data) > 0 and
                                len(mz_data) == len(int_data)):
                            continue
                        
                        # Convertir en arrays numpy et normaliser
                        ref_mz = np.array(mz_data)
                        ref_int = np.array(int_data)
                        ref_mz_norm, ref_int_norm = comparator.normalize_spectrum(ref_mz, ref_int)
                        
                        if len(ref_mz_norm) == 0:
                            continue
                        
                        # Calculer la similarité
                        similarity = comparator.calculate_cosine_similarity(
                            exp_mz_norm, exp_int_norm, ref_mz_norm, ref_int_norm
                        )
                        
                        # Garder le meilleur
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_spectrum = ref_row
                            
                            # Récupérer l'énergie de collision
                            collision_energy_clean = ref_row.get('Collision_energy_clean', None)
                            collision_energy_raw = ref_row.get('Collision_energy', None)
                            
                            if pd.notna(collision_energy_clean):
                                best_energy = f"{collision_energy_clean} eV"
                            elif pd.notna(collision_energy_raw):
                                best_energy = str(collision_energy_raw)
                            else:
                                best_energy = "inconnue"
                            
                    except Exception as e:
                        continue
                
                # Enregistrer le meilleur spectre s'il existe
                if best_spectrum is not None and best_similarity > 0:
                    best_spectra_dict[name] = {
                        'mz': best_spectrum['peaks_ms2_mz'],
                        'int': best_spectrum['peaks_ms2_intensities'],
                        'energy': best_energy,
                        'similarity': best_similarity
                    }
                    molecules_optimized += 1
            
            molecules_processed.add(name)
        
        # Initialiser les colonnes avec des valeurs par défaut
        matches_df['ms2_mz_reference'] = [[] for _ in range(len(matches_df))]
        matches_df['ms2_intensities_reference'] = [[] for _ in range(len(matches_df))]
        matches_df['collision_energy_reference'] = None
        
        # Mettre à jour ms2_similarity_score avec les nouveaux scores optimisés
        if 'ms2_similarity_score' not in matches_df.columns:
            matches_df['ms2_similarity_score'] = 0.0
        
        # Ajouter les spectres de référence optimisés
        for idx, row in matches_df.iterrows():
            molecule_name = row['match_name']
            
            if pd.notna(molecule_name) and molecule_name in best_spectra_dict:
                spectrum_info = best_spectra_dict[molecule_name]
                
                matches_df.at[idx, 'ms2_mz_reference'] = spectrum_info['mz']
                matches_df.at[idx, 'ms2_intensities_reference'] = spectrum_info['int']
                matches_df.at[idx, 'collision_energy_reference'] = spectrum_info['energy']
                matches_df.at[idx, 'ms2_similarity_score'] = spectrum_info['similarity']
            else:
                # Si pas de spectre de référence, s'assurer que les valeurs par défaut sont en place
                matches_df.at[idx, 'ms2_mz_reference'] = []
                matches_df.at[idx, 'ms2_intensities_reference'] = []
                matches_df.at[idx, 'collision_energy_reference'] = None
        
        return matches_df

    def create_feature_matrix(
        self,
        input_dir: Path,
        output_dir: Path,
        identifier: CompoundIdentifier
    ) -> None:
        """
        Crée la matrice de features et les identifications.
        """
        try:
            # 1. Alignement des features
            matrix, feature_info, raw_files = self.align_features_across_samples(input_dir)
            
            # 2. Identification et MS2
            identifications = self.process_features(feature_info, raw_files, identifier)
            
            # 3. Sauvegardes
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder la matrice d'intensités
            matrix.to_parquet(output_dir / "feature_matrix.parquet")
            matrix.to_csv(output_dir / "feature_matrix.csv")
            
            # Si on a des identifications, créer les fichiers combinés
            if not identifications.empty:
                # ÉTAPE 1: PRÉPARATION DES DATAFRAMES
                feature_info_clean = feature_info.reset_index(drop=True)
                feature_info_clean = feature_info_clean.reset_index().rename(columns={'index': 'feature_idx'})
                
                # VÉRIFICATION: S'assurer que feature_id existe
                if 'feature_id' not in feature_info_clean.columns:
                    feature_info_clean['feature_id'] = [f"F{i+1:04d}" for i in range(len(feature_info_clean))]
                
                # Nettoyer identifications
                identifications_clean = identifications.reset_index(drop=True)
                
                # ÉTAPE 2: RENOMMER LES COLONNES POUR ÉVITER LES CONFLITS
                feature_info_renamed = feature_info_clean.copy()
                if 'peaks_mz_ms2' in feature_info_renamed.columns:
                    feature_info_renamed = feature_info_renamed.rename(columns={
                        'peaks_mz_ms2': 'ms2_mz_experimental_temp',
                        'peaks_intensities_ms2': 'ms2_intensities_experimental_temp'
                    })
                
                identifications_renamed = identifications_clean.copy()
                rename_mapping = {}
                if 'ms2_mz_reference' in identifications_renamed.columns:
                    rename_mapping['ms2_mz_reference'] = 'ms2_mz_reference_temp'
                if 'ms2_intensities_reference' in identifications_renamed.columns:
                    rename_mapping['ms2_intensities_reference'] = 'ms2_intensities_reference_temp'
                if 'collision_energy_reference' in identifications_renamed.columns:
                    rename_mapping['collision_energy_reference'] = 'collision_energy_reference_temp'
                if 'ms2_similarity_score' in identifications_renamed.columns:
                    rename_mapping['ms2_similarity_score'] = 'ms2_similarity_score_temp'
                
                if rename_mapping:
                    identifications_renamed = identifications_renamed.rename(columns=rename_mapping)
                
                # Supprimer les colonnes dupliquées
                duplicate_cols = ['peaks_mz_ms2', 'peaks_intensities_ms2']
                for col in duplicate_cols:
                    if col in identifications_renamed.columns:
                        identifications_renamed = identifications_renamed.drop(columns=[col])
                
                # ÉTAPE 3: MERGE
                summary_df = pd.merge(
                    feature_info_renamed,
                    identifications_renamed,
                    on='feature_idx',
                    how='left',
                    suffixes=('_feature', '_id')
                )
                
                # ÉTAPE 4: ATTRIBUTION DES COLONNES MS2
                summary_df['ms2_mz_experimental'] = None
                summary_df['ms2_intensities_experimental'] = None
                summary_df['ms2_mz_reference'] = None
                summary_df['ms2_intensities_reference'] = None
                summary_df['collision_energy_reference'] = None
                summary_df['ms2_similarity_score'] = 0.0
                
                # Attribution des spectres expérimentaux
                if 'ms2_mz_experimental_temp' in summary_df.columns:
                    summary_df['ms2_mz_experimental'] = summary_df['ms2_mz_experimental_temp']
                    summary_df['ms2_intensities_experimental'] = summary_df['ms2_intensities_experimental_temp']
                else:
                    summary_df['ms2_mz_experimental'] = [[] for _ in range(len(summary_df))]
                    summary_df['ms2_intensities_experimental'] = [[] for _ in range(len(summary_df))]
                
                # Attribution des spectres de référence
                if 'ms2_mz_reference_temp' in summary_df.columns:
                    summary_df['ms2_mz_reference'] = summary_df['ms2_mz_reference_temp']
                    summary_df['ms2_intensities_reference'] = summary_df['ms2_intensities_reference_temp']
                else:
                    summary_df['ms2_mz_reference'] = [[] for _ in range(len(summary_df))]
                    summary_df['ms2_intensities_reference'] = [[] for _ in range(len(summary_df))]
                
                # Attribution de l'énergie de collision
                collision_energy_sources = [
                    'collision_energy_reference_temp',
                    'collision_energy_reference_feature',
                    'collision_energy_reference_id',
                    'collision_energy_reference'
                ]
                
                for source_col in collision_energy_sources:
                    if source_col in summary_df.columns:
                        non_null_mask = summary_df[source_col].notna()
                        if non_null_mask.any():
                            summary_df['collision_energy_reference'] = summary_df[source_col]
                            break
                
                # Attribution du score de similarité MS2
                similarity_sources = [
                    'ms2_similarity_score_temp',
                    'ms2_similarity_score_feature',
                    'ms2_similarity_score_id',
                    'ms2_similarity_score'
                ]
                
                for source_col in similarity_sources:
                    if source_col in summary_df.columns:
                        non_zero_mask = (summary_df[source_col].notna()) & (summary_df[source_col] > 0)
                        if non_zero_mask.any():
                            summary_df['ms2_similarity_score'] = summary_df[source_col].fillna(0.0)
                            break
                
                # ÉTAPE 5: NETTOYAGE DES COLONNES TEMPORAIRES
                cols_to_remove = []
                for col in summary_df.columns:
                    if '_temp' in col or ('_feature' in col and col != 'feature_idx') or ('_id' in col and col != 'feature_id' and col != 'molecule_id'):
                        cols_to_remove.append(col)
                
                if cols_to_remove:
                    summary_df = summary_df.drop(columns=cols_to_remove)
                
                # ÉTAPE 5.5: S'ASSURER QUE feature_id EST PRÉSERVÉ
                if 'feature_id' not in summary_df.columns:
                    # Reconstituer feature_id si perdu
                    if 'feature_idx' in summary_df.columns:
                        summary_df['feature_id'] = [f"F{i+1:04d}" for i in summary_df['feature_idx']]
                    else:
                        summary_df['feature_id'] = [f"F{i+1:04d}" for i in range(len(summary_df))]
                
                # ÉTAPE 6: VÉRIFICATIONS FINALES
                duplicate_cols = summary_df.columns[summary_df.columns.duplicated()].tolist()
                if duplicate_cols:
                    summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()]
                
                # ÉTAPE 7: SAUVEGARDES
                summary_df.to_parquet(output_dir / "features_complete.parquet", index=False)
                
                # Sauvegarder CSV avec conversion lisible des listes
                csv_df = summary_df.copy()
                list_columns = ['ms2_mz_experimental', 'ms2_intensities_experimental',
                            'ms2_mz_reference', 'ms2_intensities_reference']
                
                def format_list_for_csv(x):
                    """Convertit une liste en format lisible [val1,val2,val3] pour CSV"""
                    if isinstance(x, list) and len(x) > 0:
                        formatted_values = []
                        for item in x:
                            if isinstance(item, float):
                                formatted_values.append(f"{item:.4f}".rstrip('0').rstrip('.'))
                            elif isinstance(item, int):
                                formatted_values.append(str(item))
                            else:
                                formatted_values.append(str(item))
                        return f"[{','.join(formatted_values)}]"
                    else:
                        return "[]"
                
                for col in list_columns:
                    if col in csv_df.columns:
                        csv_df[col] = csv_df[col].apply(format_list_for_csv)
                
                csv_df.to_csv(output_dir / "features_complete.csv", index=False)
                
            else:
                # Si pas d'identifications, sauvegarder quand même les features de base
                print("   ℹ️ Aucune identification trouvée, sauvegarde des features de base")
                feature_info_clean = feature_info.reset_index(drop=True)
                
                # S'assurer que feature_id existe
                if 'feature_id' not in feature_info_clean.columns:
                    feature_info_clean['feature_id'] = [f"F{i+1:04d}" for i in range(len(feature_info_clean))]
                
                # Ajouter les colonnes manquantes pour la compatibilité avec les visualisations
                feature_info_clean['match_name'] = None
                feature_info_clean['confidence_level'] = None
                feature_info_clean['categories'] = None
                feature_info_clean['ms2_mz_experimental'] = [[] for _ in range(len(feature_info_clean))]
                feature_info_clean['ms2_intensities_experimental'] = [[] for _ in range(len(feature_info_clean))]
                feature_info_clean['ms2_mz_reference'] = [[] for _ in range(len(feature_info_clean))]
                feature_info_clean['ms2_intensities_reference'] = [[] for _ in range(len(feature_info_clean))]
                feature_info_clean['collision_energy_reference'] = None
                feature_info_clean['ms2_similarity_score'] = 0.0
                
                # S'assurer que la colonne 'samples' existe
                if 'samples' not in feature_info_clean.columns:
                    # Récupérer les noms d'échantillons de la matrice
                    sample_names = list(matrix.index)
                    feature_info_clean['samples'] = ','.join(sample_names)
                
                feature_info_clean.to_parquet(output_dir / "features_complete.parquet", index=False)
                feature_info_clean.to_csv(output_dir / "features_complete.csv", index=False)
            
            # 4. Affichage des statistiques
            self._print_feature_matrix_stats(matrix, feature_info, identifications)
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de la matrice: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
                            
    def _print_feature_matrix_stats(
        self,
        matrix: pd.DataFrame,
        feature_info: pd.DataFrame,
        identifications: pd.DataFrame
    ) -> None:
        """
        Affiche les statistiques de la matrice de features.
        
        Args:
            matrix: Matrice d'intensités
            feature_info: DataFrame des informations sur les features
            identifications: DataFrame des identifications
        """
        print("\n✅ Création de la matrice des features terminée avec succès")
        print(f"   • {matrix.shape[1]} features")
        print(f"   • {matrix.shape[0]} échantillons")
        
        if not identifications.empty:
            print("\n📊 Distribution des niveaux de confiance:")
            for sample in sorted(matrix.index):
                print(f"\n   • {sample}:")
                
                # Trouver les features présentes dans cet échantillon
                sample_features = [idx for idx, name in enumerate(matrix.columns)
                                if matrix.loc[sample, name] > 0]
                
                # Filtrer les identifications pour ces features
                sample_identifications = identifications[
                    identifications['feature_idx'].isin(sample_features)
                ]

                if len(sample_identifications) > 0:
                    for level in sorted(sample_identifications['confidence_level'].unique()):
                        level_df = sample_identifications[
                            sample_identifications['confidence_level'] == level
                        ]
                        unique_molecules = level_df['match_name'].nunique()
                        print(f"      Niveau {level}: {unique_molecules} molécules uniques")
                else:
                    print("      Aucune identification")
# Protocole Expérimental — Programme Post-Sortie Diabète

**A/B Test : Réduction des Réadmissions < 30 Jours**

_Version 1.0 — Projet Diabetic 130-US_

---

## 1. Contexte & Objectif

Le modèle de prédiction (notebook 05) identifie les patients à risque élevé de réadmission dans les 30 jours suivant leur sortie. Le taux de réadmission `<30` observé sur le dataset de référence est de **11,3 %** (baseline). Ce protocole vise à valider expérimentalement qu'un programme post-sortie ciblé réduit ce taux de façon statistiquement significative et économiquement rentable.

**Hypothèse principale :** un programme d'accompagnement post-sortie (appel J+2, téléconsultation J+7, coordination du médecin traitant) réduit le taux de réadmission `<30` d'au moins 15 % en relatif chez les patients scorés dans le top 10 % de risque.

---

## 2. Design Expérimental

### 2.1 Population cible

- **Critères d'inclusion :** tout patient hospitalisé pour diabète, scoré dans le **top 10 % de risque** selon le modèle RandomForest (P(`<30`) ≥ percentile 90 de la population sortante du jour).
- **Critères d'exclusion :** décès durant le séjour, transfert vers réanimation, refus explicite du patient, absence de numéro de contact valide.

### 2.2 Groupes et randomisation

| Groupe | Désignation | Prise en charge                                    |
| ------ | ----------- | -------------------------------------------------- |
| **A**  | Contrôle    | Soins standard (aucune intervention additionnelle) |
| **B**  | Traitement  | Soins standard + programme post-sortie             |

- **Unité de randomisation : le patient** (et non le séjour), afin d'éviter toute contamination en cas de ré-hospitalisation dans la période d'observation.
- **Ratio :** 1:1, allocation par blocs permutés de taille 4 (séquence générée à J−1 du démarrage, conservée sous scellé).
- **Masquage :** ouvert côté soignant ; l'analyste principal est en aveugle jusqu'à l'analyse finale.

### 2.3 Description de l'intervention (Groupe B)

| Étape               | Délai            | Responsable              | Contenu                                                                       |
| ------------------- | ---------------- | ------------------------ | ----------------------------------------------------------------------------- |
| Appel de suivi      | J+2 après sortie | Infirmière coordinatrice | Vérification de l'état général, observance médicamenteuse, questions ouvertes |
| Téléconsultation    | J+7              | Médecin référent         | Bilan clinique, révision ordonnance si poly-médication (≥ 12 médicaments)     |
| Coordination MT     | J+3 à J+5        | Infirmière coordinatrice | Transmission du résumé de sortie, confirmation du RDV MT sous 7 jours         |
| Plan de soins écrit | Jour de sortie   | Soignant de sortie       | Document simplifié : signes d'alarme, traitements, contacts d'urgence         |

---

## 3. Calcul de la Taille d'Échantillon

**Hypothèses retenues** (MDE = −15 % en relatif, scénario réaliste au regard de la littérature) :

| Paramètre            | Valeur                |
| -------------------- | --------------------- |
| Taux baseline `p0`   | 11,3 %                |
| Taux cible `p1`      | 9,6 % (−15 % relatif) |
| Risque α (bilatéral) | 5 %                   |
| Puissance (1 − β)    | 80 %                  |
| **n par bras**       | **5 366 patients**    |
| **Total**            | **10 732 patients**   |
| Durée estimée\*      | ~22 semaines          |

_\* Hypothèse : ~500 sorties éligibles / semaine dans le périmètre de déploiement._

**Analyse de sensibilité sur le MDE :**

| MDE relatif | p1        | n / bras  | Durée estimée               |
| ----------- | --------- | --------- | --------------------------- |
| −10 %       | 10,2 %    | 12 157    | ~49 semaines _(non retenu)_ |
| **−15 %**   | **9,6 %** | **5 366** | **~22 semaines ✅**         |
| −20 %       | 9,0 %     | 2 997     | ~12 semaines                |

---

## 4. Métriques & Analyse Statistique

### 4.1 Métrique primaire

**Taux de réadmission toutes causes `<30` jours** — comparé entre les groupes A et B par un **z-test bilatéral sur proportions** (analyse en intention de traiter, ITT).

Seuil de décision : **p < 0,05**, avec calcul de l'intervalle de confiance à 95 % sur la différence absolue et relative.

### 4.2 Métriques secondaires

- Taux de réadmission `>30` jours (impact à moyen terme).
- Nombre de passages aux urgences dans les 30 jours post-sortie.
- Coût cumulé par patient (soins + programme) sur 30 jours.
- Taux d'adhérence au programme (Groupe B uniquement) : % de patients ayant reçu les 3 étapes.

### 4.3 Guardrails (arrêt anticipé si franchis)

- Mortalité toutes causes à 30 jours significativement supérieure dans le Groupe B.
- Taux d'incidents liés à l'intervention > 2 % (hospitalisations iatrogènes, effets indésirables signalés).
- Surconsommation d'actes (consultations non planifiées) supérieure de 30 % dans le Groupe B sans bénéfice clinique associé.

---

## 5. Sous-analyses Pré-spécifiées

Toutes les sous-analyses ci-dessous sont **pré-enregistrées** avant le démarrage et analysées sur le même critère primaire :

1. **Par cluster de risque** (notebook 03) : vérifier si l'effet est homogène ou concentré sur le cluster _Polypathologique chronique_ (interaction attendue selon la littérature).
2. **Par tranche d'âge** : ≤ 60 ans / 61–75 ans / > 75 ans.
3. **Par modalité de sortie** : domicile sans suivi (code 1) vs. SSR / réhabilitation (codes 3–4).
4. **Par type d'admission** : urgences vs. programmé.

Les p-values des sous-analyses seront corrigées par la méthode de Benjamini-Hochberg (contrôle du FDR à 10 %).

---

## 6. Calendrier & Gouvernance

| Jalon                                      | Délai             |
| ------------------------------------------ | ----------------- |
| Validation du protocole (comité éthique)   | J0                |
| Démarrage recrutement                      | J0 + 4 semaines   |
| Analyse intermédiaire (50 % de l'effectif) | ~J0 + 13 semaines |
| Fin du recrutement                         | ~J0 + 26 semaines |
| Analyse finale & rapport                   | J0 + 28 semaines  |
| Comité médico-économique de décision       | J0 + 30 semaines  |

**Analyse intermédiaire :** réalisée à 50 % de l'effectif par un statisticien indépendant. Arrêt pour futilité si OR < 1,05 ; arrêt pour efficacité si p < 0,001 (règle d'O'Brien-Fleming).

---

## 7. Critère de Déploiement

Le programme sera déployé à l'ensemble de la population éligible si et seulement si **les 3 conditions suivantes** sont simultanément réunies à l'issue de l'analyse finale :

1. Réduction du taux `<30` statistiquement significative (**p < 0,05**, ITT).
2. Gain net financier positif sur la période d'observation (économies − coût programme > 0).
3. Aucun guardrail franchi.

La décision sera formalisée en **comité médico-économique** réunissant la direction médicale, la direction financière et le référent data, à J+30 après la fin du recrutement.

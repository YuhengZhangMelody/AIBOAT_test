# Résumé des modifications EKF-SLAM pour GNSS

## 📝 Ce qui a été changé

### 1. **Initialisation (`__init__`)**
- ✅ Ajout de `self.USE_GNSS = True` pour contrôler l'activation/désactivation
- ✅ Ajout de `self.R_gnss` : matrice de covariance du bruit GNSS (2×2)

```python
self.USE_GNSS = True  # Enable/disable GNSS correction
self.R_gnss = np.diag([1.0, 1.0]) ** 2  # Default: 1m std.dev.
```

### 2. **Nouvelle fonction : `calc_innovation_gnss(xGnss)`**
- ✅ Calcule l'innovation pour les mesures GNSS
- ✅ Retourne: innovation, covariance d'innovation, Jacobien
- Modèle d'observation simple: position absolue [x, y]

```python
def calc_innovation_gnss(self, xGnss):
    """Compute innovation for GNSS position measurement"""
    innov = xGnss - xp  # measurement - predicted
    H = [1, 0, 0, 0, ...]  # Only affects x, y
    S = H @ P @ H^T + R_gnss
    return innov, S, H
```

### 3. **Modification : `ekf_slam(u, y, xGnss=None)`**
- ✅ Signature modifiée pour accepter mesures GNSS (optionnel)
- ✅ Ajoute une phase d'update GNSS après les updates landmarks
- ✅ GNSS est facultatif (xGnss=None signifie pas de mesure GNSS ce pas)

Ordre d'exécution:
```
1. Prediction (avec u)
2. Update landmarks (avec y)
3. Update GNSS (avec xGnss) ← NOUVEAU
```

### 4. **Modification : `get_estimate_full_motion(uTrue, xGnss=None)`**
- ✅ Signature modifiée pour passer xGnss au SLAM
- ✅ Permet d'utiliser le GNSS dans le workflow existant

## 🔧 Comment utiliser

### Cas 1 : Sans GNSS (comportement original)
```python
xEst, PEst = slam.get_estimate_full_motion(u)
```

### Cas 2 : Avec GNSS
```python
xGnss = np.array([[gps_x], [gps_y]])  # Position GNSS [2×1]
xEst, PEst = slam.get_estimate_full_motion(u, xGnss=xGnss)
```

### Cas 3 : Contrôle fine-grained
```python
slam.USE_GNSS = True  # ou False
slam.R_gnss = np.diag([2.0, 2.0]) ** 2  # Tuner le bruit

if gnss_data_available:
    xEst, PEst = slam.ekf_slam(u, y, xGnss)
else:
    xEst, PEst = slam.ekf_slam(u, y, xGnss=None)
```

## ⚠️ Paramètres à ajuster pour votre bateau

### `slam.R_gnss` : CRITIQUE
La matrice de covariance du bruit GNSS doit refléter la précision réelle de votre système.

| Scenario | R_gnss | Quand l'utiliser |
|----------|--------|------------------|
| GPS de test/budget | `diag([5.0, 5.0])**2` | 5m d'erreur standard |
| GPS standard | `diag([2.0, 2.0])**2` | 2m d'erreur standard (typique) |
| GPS précis (RTK) | `diag([0.1, 0.1])**2` | RTK-corrected, <10cm |
| Variables selon la région | `diag([var_x, var_y])**2` | Si x plus précis que y |

**Comment déterminer la bonne valeur:**
1. Enregistrer les mesures GPS quand le bateau est stationnaire
2. Calculer écart-type: `std_x = np.std(gps_x - mean_gps_x)`
3. Mettre `R_gnss = diag([std_x, std_y])**2`

## 📊 Conceptual Diagram

```
┌─────────────────────────────────────────────────┐
│         EKF-SLAM Main Loop                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. PREDICTION PHASE                            │
│     x_pred = f(x_est, u)                        │
│     P_pred = A @ P @ A^T + B @ Q @ B^T         │
│                                                 │
│  2. LANDMARK UPDATE                             │
│     FOR each landmark observation y:            │
│         innov = y - h_landmark(x)               │
│         K = P @ H^T @ inv(S)                   │
│         x = x + K @ innov                      │
│         P = (I - K @ H) @ P                    │
│                                                 │
│  3. GNSS UPDATE (NEW)  ← ← ← ← ← ← ← ← ← ←   │
│     IF xGnss is not None AND USE_GNSS:         │
│         innov = xGnss - x[0:2]                 │
│         H_gnss = [1, 0, 0, ...; 0, 1, 0, ...] │
│         S = H_gnss @ P @ H_gnss^T + R_gnss    │
│         K_gnss = P @ H_gnss^T @ inv(S)        │
│         x = x + K_gnss @ innov                 │
│         P = (I - K_gnss @ H_gnss) @ P         │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 🎯 Avantages de la fusion GNSS

| Bénéfice | Explication |
|----------|------------|
| **Robustesse positionnelle** | Position absolue pour éviter dérive accumulative |
| **Observabilité position** | Le GNSS observe directement [x,y], pas via triangulation |
| **Convergence rapide** | Moins d'hypothèses sur correspondance landmarks |
| **Fallback intelligent** | Si GNSS échoue → SLAM continue (avec landmarks) |
| **Validation croisée** | Landmarks valident/corrigent GNSS et vice-versa |

## ⚠️ Pièges courants

### ❌ Erreur 1 : R_gnss beaucoup trop petit
```python
slam.R_gnss = np.diag([0.01, 0.01])**2  # Mauvais! Dit que GPS est parfait
# Résultat: Le filtre croit aveuglément le GPS, landmarks ignorés
```

### ❌ Erreur 2 : R_gnss beaucoup trop grand
```python
slam.R_gnss = np.diag([100.0, 100.0])**2  # Trop grand
# Résultat: Le filtre ignore le GPS, pas d'amélioration
```

### ❌ Erreur 3 : Mesures GNSS avec sauts
```python
# GPS avec multi-path ou perte de satellite
xGnss = np.array([[position_gps_x], [position_gps_y]])  
# Peut avoir saut de 10m...
# Solution: Implémenter outlier detection
```

### ✅ Solution : Test d'innovation
```python
innov_norm = np.linalg.norm(innovation)
if innov_norm > 3 * np.sqrt(S.trace()):
    # Innovation too large, reject GNSS
    print(f"GNSS rejected, innovation too large: {innov_norm:.2f}m")
else:
    # Accept GNSS update
    x = x + K @ innov
```

## 📋 Checklist d'implémentation

- [ ] Déterminer la précision réelle de votre GPS (erreur standard)
- [ ] Configurer `slam.R_gnss` selon cette précision
- [ ] Tester le SLAM **sans** GNSS d'abord (baseline)
- [ ] Ajouter données GNSS progressivement
- [ ] Observer l'amélioration de la covariance
- [ ] Mettre en place détection d'anomalies GNSS
- [ ] Logger innovation et covariance pour analyse
- [ ] Valider dans votre scénario d'utilisation (eau, obstacles, etc.)

## 📚 Références sur la fusion capteurs

1. **EKF multi-capteur**: Standard en robotique (Thrun "Probabilistic Robotics")
2. **GPS + IMU**: Classique en navigation (Grewal & Andrews *Kalman Filtering*)
3. **GPS + Vision**: Appliqué en robotique marine
4. **Outlier rejection**: Test d'innovation (Mahalanobis distance)

## 🤝 Questions fréquentes

**Q: Dois-je garder les landmarks si j'utilise le GNSS?**
R: Oui! Les deux fournissent des informations complémentaires:
- GNSS: Position absolue mais bruitée
- Landmarks: Position relative mais précise (si bien détectés)

**Q: Comment gérer un GPS qui s'allume/éteint?**
R: Utiliser la signature modifiée:
```python
if gnss_available:
    xEst, PEst = slam.ekf_slam(u, y, xGnss)
else:
    xEst, PEst = slam.ekf_slam(u, y, xGnss=None)
```

**Q: Peut-on utiliser plusieurs GPSs?**
R: Oui, en créant plusieurs updates type GNSS dans la boucle (ou plutôt appeler une fonction spécifique).

---

**Créé**: 2026-02-09  
**Fichier modifié**: `ekf_slam.py`  
**Fichiers ajoutés**: `GNSS_INTEGRATION_GUIDE.md`, `example_gnss_usage.py`, `TESTING_CHECKLIST.md`

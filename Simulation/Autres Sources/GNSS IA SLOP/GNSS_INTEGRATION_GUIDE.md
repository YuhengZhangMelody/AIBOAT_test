# Guide d'Intégration GNSS dans le SLAM

## 📋 Résumé de la modification

Vous avez ajouté support **optionnel** du GNSS à votre EKF-SLAM. Les modifications permettent d'utiliser les mesures GNSS **en parallèle** avec les observations de landmarks, pas en remplacement.

## ✅ Pourquoi cette approche est meilleure

### Ce que vous aviez proposé ❌
Ajouter GNSS directement dans l'innovation des landmarks :
- **Problème** : L'innovation est la différence entre observation *prédite* et observation *mesurée*
- **Incohérence** : Landmarks donnent (distance, angle), GNSS donne (x, y) absolus
- **Jacobiens incompatibles** : Les modèles d'observation sont fondamentalement différents

### Ce que vous avez maintenant ✅
Traiter GNSS comme une **observation indépendante** :
- **Modèle d'observation GNSS** : Simple, c'est une mesure absolue de position
  $$z_{GNSS} = \begin{bmatrix} x \\ y \end{bmatrix} + v_{GNSS}$$

- **Jacobien GNSS** : Simple (identité sur les positions)
  $$H_{GNSS} = \begin{bmatrix} 1 & 0 & 0 & 0 & ... \\ 0 & 1 & 0 & 0 & ... \end{bmatrix}$$

- **Fusion sensorielle correcte** : Chaque capteur garde sa propre représentation

## 🔧 Comment utiliser dans votre code

### Utilisation basique

```python
# Dans votre boucle principale (main_dwa.py, par exemple)
from ekf_slam import Slam

slam = Slam(Landmarks, initial_pose, boat)

# À chaque itération
xEst, PEst = slam.get_estimate_full_motion(u_control)  # Sans GNSS

# Avec GNSS
xGnss = np.array([[x_gnss], [y_gnss]])  # Position GNSS en mètres [2x1]
xEst, PEst = slam.get_estimate_full_motion(u_control, xGnss=xGnss)
```

### Contrôle du GNSS

```python
slam.USE_GNSS = True   # Activer GNSS
slam.USE_GNSS = False  # Désactiver GNSS (ne l'utilise que les landmarks)
```

### Tuning du bruit GNSS

```python
# Par défaut (1m d'écart-type pour x et y)
slam.R_gnss = np.diag([1.0, 1.0]) ** 2

# Si votre GPS est moins précis (5m d'écart-type)
slam.R_gnss = np.diag([5.0, 5.0]) ** 2

# Si votre GPS est très précis (0.5m d'écart-type)
slam.R_gnss = np.diag([0.5, 0.5]) ** 2

# Ou des bruits différents en x et y
slam.R_gnss = np.diag([1.0, 2.0]) ** 2  # 1m en x, 2m en y
```

## 📊 Ajustements à faire

### 1. **Estimer la précision de votre GNSS**

Avant de régler `R_gnss`, déterminez le bruit réel de votre GPS :

```
R_gnss = (écart-type du GPS) ² 
```

Par exemple, si le GPS a ~2m d'erreur :
```python
slam.R_gnss = np.diag([2.0, 2.0]) ** 2  # = [[4, 0], [0, 4]]
```

### 2. **Gérer le manque de signal GNSS**

```python
# Faire un test avant d'appeler

if gnss_signal_valid and xGnss is not None:
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
else:
    # Sans GNSS : le SLAM dépendra uniquement des landmarks
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss=None)
```

### 3. **Fusion progressive (recommandé)**

Si vous doutez initialement de la qualité du GNSS, vous pouvez commencer sans lui :

```python
# Phase d'initialisation (utiliser landmarks uniquement)
if not initialized_with_landmarks:
    slam.USE_GNSS = False
    initialize(slam, ...)
    initialized_with_landmarks = True

# Phase opérationnelle (ajouter GNSS)
slam.USE_GNSS = True
slam.get_estimate_full_motion(u, xGnss)
```

## 🎯 Avantages de cette approche

| Aspect | Détail |
|--------|--------|
| **Robustesse** | Si GNSS se perd → SLAM continue avec landmarks |
| **Flexibilité** | On/off facile, paramétrage indépendant |
| **Convergence** | Deux sources d'information → convergence plus rapide |
| **Observabilité** | Position absolute (GNSS) + relative (landmarks) = meilleure estimation |
| **Traçabilité** | Chaque capteur gère son bruit indépendamment |

## ⚠️ Points d'attention

### 1. **Place des landmarks en cas de GPS parfait**

Si vous aviez un GPS parfait (erreur = 0), les landmarks deviendraient "inutiles" pour la position. Ils aideraient surtout si :
- GPS se perd = landmarks prennent le relais
- Améliorent la covariance globale
- Permettent une vérification croisée

### 2. **Ordre des updates**

Actuellement, vous appliquez :
1. Updates landmarks
2. Update GNSS

Cet ordre importe peu pour la convergence finale, mais peut affecter la trajectoire transitoire.

### 3. **Consistency du filtre**

⚠️ **Important** : Vérifiez que votre filtre reste "consistent" (la covariance estimée reflète vraiment l'erreur réelle).

Signes d'inconsistency :
- La covariance diminue trop vite → vous surestimez la précision
- La covariance n'améliore pas → le filtre ignore les mesures

## 📈 Améliorations futures possibles

### 1. **Adaptive noise tuning**
Estimer `R_gnss` automatiquement selon la convergence :
```python
if innovation_gnss_too_large:
    slam.R_gnss *= 1.2  # Relaxer l'hypothèse de précision
else:
    slam.R_gnss *= 0.99  # Augmenter légèrement la confiance
```

### 2. **Détection d'anomalies GNSS**
```python
# Rejeter les mesures GNSS aberrantes
innovation_threshold = 10  # m
if np.linalg.norm(innov_gnss) > innovation_threshold:
    print("GNSS measurement rejected")
    # Ne pas appliquer update GNSS
```

### 3. **Fusion IMU**
Ajouter des données IMU pour un meilleur modèle de mouvement et améliorer la prédiction.

### 4. **Multi-hypothesis SLAM**
Gérer plusieurs hypothèses de correspondance de landmarks (déjà partiellement là).

## 📝 Exemple complet d'utilisation

```python
#!/usr/bin/env python
import numpy as np
from ekf_slam import Slam
from boat_state import Boat

# Initialisation
landmarks = np.array([[10, 10], [20, 5], [15, 20]])
initial_pose = np.array([[0], [0], [0]])
boat = Boat()
slam = Slam(landmarks, initial_pose, boat)

# Configurer GNSS
slam.USE_GNSS = True
slam.R_gnss = np.diag([2.0, 2.0]) ** 2  # 2m de bruit GPS

# Simulation
for step in range(100):
    # Commande de contrôle
    u = np.array([[1.0], [0.1]])  # [vitesse, angle_braquage]
    
    # Mesure GNSS (à remplacer par votre source réelle)
    xGnss = np.array([[?], [?]])  # À obtenir de votre capteur GPS
    
    # Estimation SLAM
    if xGnss is not None:
        xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    else:
        xEst, PEst = slam.get_estimate_full_motion(u)
    
    # Afficher résultats
    print(f"Step {step}: pos=({xEst[0,0]:.2f}, {xEst[1,0]:.2f}), yaw={np.rad2deg(xEst[2,0]):.1f}°")
```

## 🤔 FAQ

**Q: Dois-je retirer les landmarks si j'ai le GNSS?**  
R: Non! Les deux sources d'information sont complémentaires. Les landmarks aident à la robustesse.

**Q: Que faire si le GNSS a des sauts soudains?**  
R: Implémenter un test d'innovation (voir section "Détection d'anomalies").

**Q: La covariance GNSS doit-elle être la même que celle du filtre?**  
R: Non, `R_gnss` c'est le bruit de *mesure* GNSS, indépendant de l'état du filtre.

**Q: Peut-on utiliser GNSS sans landmarks?**  
R: Techniquement oui, mais ce serait du filtre de Kalman simple, pas du SLAM.

---

**Auteur**: Notes d'intégration GNSS pour IABoat  
**Date**: 2026-02-09

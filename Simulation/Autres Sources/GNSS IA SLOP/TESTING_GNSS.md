# 🧪 GNSS Integration Testing Checklist

## Phase 1: Validation du code

- [ ] **Imports** : Vérifier que numpy est importé
- [ ] **Syntaxe** : `python -m py_compile ekf_slam.py` réussit
- [ ] **Classes** : Vérifier que les nouvelles méthodes existent
  ```python
  assert hasattr(slam, 'calc_innovation_gnss')
  assert hasattr(slam, 'R_gnss')
  assert hasattr(slam, 'USE_GNSS')
  ```

## Phase 2: Test unitaire simple

Créer un script test simple:

```python
import numpy as np
from ekf_slam import Slam

# Minimal test
landmarks = np.array([[10, 10], [20, 5]])
initial_pose = np.array([[0], [0], [0]])

# Créer SLAM sans bateau (si possible)
try:
    slam = Slam(landmarks, initial_pose, MockBoat())
except Exception as e:
    print(f"Error creating SLAM: {e}")
    
# Test GNSS
xGnss = np.array([[1.0], [1.0]])
try:
    innov, S, H = slam.calc_innovation_gnss(xGnss)
    print(f"✓ GNSS innovation calculated: {innov.shape}")
except Exception as e:
    print(f"✗ Error in calc_innovation_gnss: {e}")
```

## Phase 3: Test du bruit GNSS

### 3.1 Vérifier les dimensions
```python
# R_gnss doit être 2×2
assert slam.R_gnss.shape == (2, 2), f"Expected (2,2), got {slam.R_gnss.shape}"
print(f"✓ R_gnss shape: {slam.R_gnss.shape}")

# R_gnss doit être positif défini
eigvals = np.linalg.eigvalsh(slam.R_gnss)
assert np.all(eigvals > 0), f"R_gnss not positive definite: {eigvals}"
print(f"✓ R_gnss is positive definite: {eigvals}")
```

### 3.2 Tester différentes valeurs de bruit
```python
test_cases = [
    (0.5, "RTK GPS"),       # Très précis
    (1.0, "Standard GPS"),  # Moderate
    (5.0, "Poor GPS"),      # Mauvais signal
]

for std_dev, description in test_cases:
    slam.R_gnss = np.diag([std_dev, std_dev]) ** 2
    print(f"Testing {description} (σ={std_dev}m):")
    print(f"  R_gnss diagonal: {np.diag(slam.R_gnss)}")
```

## Phase 4: Test avec landmarks et GNSS

### 4.1 Scénario 1: GNSS seul
```python
# Créer SLAM
slam.USE_GNSS = True
slam.USE_LANDMARKS = False  # (si possible)

# Pas de landmarks
y = np.array([]).reshape(0, 3)

# Commande
u = np.array([[1.0], [0.1]])

# Mesure GNSS
xGnss = np.array([[1.0], [1.0]])

# Appeler SLAM
try:
    xEst, PEst = slam.ekf_slam(u, y, xGnss)
    print(f"✓ SLAM with GNSS only works")
    print(f"  Estimated position: ({xEst[0,0]:.3f}, {xEst[1,0]:.3f})")
except Exception as e:
    print(f"✗ Error with GNSS only: {e}")
```

### 4.2 Scénario 2: Landmarks + GNSS
```python
# Avec landmarks + GNSS
y = np.array([[1.0, np.deg2rad(45), 0]])  # Une observation

xGnss = np.array([[1.5], [0.5]])

try:
    xEst, PEst = slam.ekf_slam(u, y, xGnss)
    print(f"✓ SLAM with landmarks+GNSS works")
except Exception as e:
    print(f"✗ Error with landmarks+GNSS: {e}")
```

### 4.3 Scénario 3: Sans GNSS (backward compatibility)
```python
# Vérifier backward compatibility
try:
    xEst, PEst = slam.ekf_slam(u, y, xGnss=None)
    print(f"✓ SLAM without GNSS works (backward compatible)")
except Exception as e:
    print(f"✗ Error without GNSS: {e}")
```

## Phase 5: Test de convergence

### 5.1 Trajectoire simple
```python
import numpy as np

# Simulation simple: bateau en ligne droite
slam.USE_GNSS = True
slam.R_gnss = np.diag([2.0, 2.0]) ** 2

trajectory_true = []
trajectory_est = []
trajectory_gnss = []

for step in range(50):
    # Commande constante
    u = np.array([[1.0], [0.0]])  # 1 m/s, pas de rotation
    
    # GNSS avec bruit
    x_true, y_true = step * 1.0, 0.0
    gnss_noise = np.random.randn(2) * 2.0
    xGnss = np.array([[x_true + gnss_noise[0]], 
                      [y_true + gnss_noise[1]]])
    
    # SLAM update
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    
    trajectory_true.append([x_true, y_true])
    trajectory_est.append([xEst[0,0], xEst[1,0]])
    trajectory_gnss.append([xGnss[0,0], xGnss[1,0]])

# Analyser convergence
trajectory_true = np.array(trajectory_true)
trajectory_est = np.array(trajectory_est)
trajectory_gnss = np.array(trajectory_gnss)

error_gnss = np.linalg.norm(trajectory_true - trajectory_gnss, axis=1)
error_est = np.linalg.norm(trajectory_true - trajectory_est, axis=1)

print(f"\nConvergence Test:")
print(f"GNSS mean error:  {np.mean(error_gnss):.4f} m")
print(f"SLAM mean error:  {np.mean(error_est):.4f} m")
print(f"Improvement:      {(np.mean(error_gnss) - np.mean(error_est)) / np.mean(error_gnss) * 100:.1f}%")

if np.mean(error_est) < np.mean(error_gnss):
    print("✓ SLAM improves over raw GNSS")
else:
    print("⚠ SLAM doesn't improve over GNSS (check tuning)")
```

## Phase 6: Test de robustesse

### 6.1 Test avec GNSS absent
```python
# Simuler perte de GNSS
num_with_gnss = 30
num_without_gnss = 20

for step in range(num_with_gnss):
    u = np.array([[1.0], [0.0]])
    xGnss = np.array([[step * 1.0 + np.random.randn() * 2.0],
                      [np.random.randn() * 2.0]])
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)

print("✓ Phase 1: With GNSS completed")

# Perte du signal GPS
for step in range(num_without_gnss):
    u = np.array([[1.0], [0.0]])
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss=None)

print("✓ Phase 2: Without GNSS completed (fallback to landmarks)")

# Récupération du signal
for step in range(10):
    u = np.array([[1.0], [0.0]])
    xGnss = np.array([[step * 1.0],
                      [0.0]])
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)

print("✓ Phase 3: GNSS recovery completed")
```

### 6.2 Test avec outliers GNSS
```python
# Simuler des mesures aberrantes
for step in range(50):
    u = np.array([[1.0], [0.0]])
    
    # Ajouter une mesure aberrante toutes les 10 itérations
    if step % 10 == 5:
        # Saut de 20m (multi-path ou perte satellite)
        xGnss = np.array([[step * 1.0 + 20.0],
                          [10.0]])
        print(f"  Step {step}: Outlier GNSS")
    else:
        xGnss = np.array([[step * 1.0 + np.random.randn() * 2.0],
                          [np.random.randn() * 2.0]])
    
    try:
        xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    except np.linalg.LinAlgError:
        print(f"⚠ Numerical instability at step {step}")

print("✓ Outlier handling test completed")
```

## Phase 7: Analyse matricielle

### 7.1 Vérifier Jacobien GNSS
```python
# Le Jacobien GNSS doit être [2 × n_state]
slam.xEst = np.array([[1.0], [2.0], [0.5]] + [[3.0], [4.0]] * 2)  # 2 landmarks

xGnss = np.array([[1.5], [2.5]])
innov, S, H = slam.calc_innovation_gnss(xGnss)

print(f"\nJacobian GNSS Analysis:")
print(f"  Shape: {H.shape}")
assert H.shape == (2, len(slam.xEst)), f"H shape mismatch"
print(f"✓ H shape correct")

# H doit avoir 1 en positions [0,0] et [1,1]
assert H[0, 0] == 1.0, "H[0,0] should be 1.0"
assert H[1, 1] == 1.0, "H[1,1] should be 1.0"
print(f"✓ H has correct identity pattern for position")

# S doit être 2×2 et symétrique
assert S.shape == (2, 2), f"S shape should be (2,2), got {S.shape}"
assert np.allclose(S, S.T), "S should be symmetric"
print(f"✓ S is 2×2 and symmetric")
```

### 7.2 Vérifier gain de Kalman
```python
# Kalman gain doit avoir shape [n_state, 2]
K_gnss = slam.PEst @ H.T @ np.linalg.inv(S)

print(f"\nKalman Gain Analysis:")
print(f"  Shape: {K_gnss.shape}")
assert K_gnss.shape == (len(slam.xEst), 2), "K shape mismatch"
print(f"✓ K shape correct")

# Vérifier que K n'est pas NaN
assert not np.any(np.isnan(K_gnss)), "K contains NaN"
print(f"✓ K contains no NaN")
```

## Phase 8: Comparaison avec/sans GNSS

```python
# Run SLAM avec et sans GNSS, comparer résultats
results = {
    'without_gnss': [],
    'with_gnss': [],
}

# Test 1: Sans GNSS
slam.USE_GNSS = False
for step in range(30):
    u = np.array([[1.0], [0.0]])
    xEst, PEst = slam.get_estimate_full_motion(u)
    results['without_gnss'].append(np.sqrt(slam.PEst[0,0]**2 + slam.PEst[1,1]**2))

# Test 2: Avec GNSS
slam.USE_GNSS = True
for step in range(30):
    u = np.array([[1.0], [0.0]])
    xGnss = np.array([[step], [0]]) + np.random.randn(2, 1) * 2.0
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    results['with_gnss'].append(np.sqrt(slam.PEst[0,0]**2 + slam.PEst[1,1]**2))

# Compare
uncertainty_without = np.mean(results['without_gnss'])
uncertainty_with = np.mean(results['with_gnss'])

print(f"\nUncertainty Reduction:")
print(f"  Without GNSS: {uncertainty_without:.4f}")
print(f"  With GNSS:    {uncertainty_with:.4f}")
print(f"  Reduction:    {(uncertainty_without - uncertainty_with) / uncertainty_without * 100:.1f}%")

if uncertainty_with < uncertainty_without:
    print("✓ GNSS reduces position uncertainty")
else:
    print("⚠ GNSS does not reduce uncertainty (check tuning)")
```

## 📋 Checklist finale

- [ ] Phase 1: Code valide
- [ ] Phase 2: Fonctions GNSS accessibles
- [ ] Phase 3: Matrice de bruit GNSS correcte
- [ ] Phase 4: GNSS seul fonctionne
- [ ] Phase 4: Landmarks + GNSS fonctionne
- [ ] Phase 4: Backward compatibility OK
- [ ] Phase 5: Convergence observée
- [ ] Phase 6: Robustesse sans GNSS
- [ ] Phase 6: Gestion des outliers OK
- [ ] Phase 7: Dimensions matricielles correctes
- [ ] Phase 7: Gain de Kalman valide
- [ ] Phase 8: GNSS réduit l'incertitude

## ✅ Si tous les tests passent:

Votre intégration GNSS est **prête pour la production** !

## ❌ Si des tests échouent:

Vérifier dans cet ordre:
1. Dimensions des matrices
2. Valeur de `R_gnss` (pas trop petit, pas trop grand)
3. Utiliser debugger pour tracer l'exécution
4. Checker les logs d'innovation pour outliers

---

**Créé**: 2026-02-09  
**Pour**: Projet IABoat  
**Durée estimée**: 1-2 heures de testing complet

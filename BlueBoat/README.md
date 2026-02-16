# BlueBoat

Code pour faire fonctionner le BlueBoat depuis une Jetson : communication, interface, vision, et futur mode autonome.

## Dossiers

### Samples
Tests de base :
- Télémétrie BlueBoat  
- Commandes moteurs simples  
- Validation du lien MAVLink  

### VisionMode
Ajoute la vision embarquée :
- Caméras connectées à la Jetson  
- Détection d’objets  
- Estimation de profondeur  

### AutonomousMode
(En développement)
- Navigation autonome  
- Évitement d’obstacles  
- Suivi de trajectoire  

## Environnement requis

Le code doit être exécuté sur une **NVIDIA Jetson** configurée avec :

- Le **ZED SDK** installé  
- L’API Python du SDK (`pyzed`) fonctionnelle  

Installation des dépendances Python :

```bash
pip install -r requirements.txt
```

## Configuration réseau Jetson ↔ BlueBoat

Pour établir la connexion Ethernet avec le BlueBoat, configurer l’interface réseau de la Jetson :

```bash
sudo ip addr flush dev eno1
sudo ip addr add 192.168.2.1/24 dev eno1
sudo ip link set eno1 up
```

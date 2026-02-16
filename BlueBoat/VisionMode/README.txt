Pour établir une connexion avec le blueboat il faut configurer le réseau éthernet de la JETSON via les commandes suivantes : 

`sudo ip addr flush dev eno1
sudo ip addr add 192.168.2.1/24 dev eno1
sudo ip link set eno1 up`


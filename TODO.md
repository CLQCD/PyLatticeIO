# Design

## LatticeSpinColorVector
### LatticeFermion
1; Lt, Lz, Ly, Lx, Ns, Nc

### MultiLatticeFermion
L5; Lt, Lz, Ly, Lx, Ns, Nc

## LatticeSpinColorMatrix
### LatticePropagator
1; Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc

## LatticeColorVector
### LatticeStaggeredFermion
1; Lt, Lz, Ly, Lx, Nc

### MultiLatticeStaggeredFermion
L5; Lt, Lz, Ly, Lx, Nc

### LatticeEigenvector
Ne; Lt, Lz, Ly, Lx, Nc

## LatticeColorMatrix
### LatticeStaggeredPropagator
1; Lt, Lz, Ly, Lx, Nc, Nc

### LatticeGauge
Nd; Lt, Lz, Ly, Lx, Nc, Nc

## EigenMatrix
### LatticeElemental
Lt; Ne, Ne

## SpinEigenMatrix
### LatticePerambulator
Lt, Lt; Ns, Ns, Ne, Ne

"""Psi4 harmonic frequencies and IR spectrum; missing intensities raise an error."""
import numpy as np
from common import *

def main():
    p=parser(__doc__); quantum_args(p)
    p.add_argument('--scale',type=positive,default=1.0)
    p.add_argument('--fwhm',type=positive,default=12.0,help='cm^-1')
    p.add_argument('--skip-optimisation',action='store_true')
    a=p.parse_args(); psi4,mol,out=psi4_setup(a)
    if not a.skip_optimisation: psi4.optimize(a.method,molecule=mol)
    energy,wfn=psi4.frequency(a.method,molecule=mol,return_wfn=True)
    analysis=wfn.frequency_analysis
    if 'IR_intensity' not in analysis:
        raise RuntimeError('This Psi4 method/version did not provide dipole derivatives/IR intensities. Inspect psi4.log; do not substitute unit intensities.')
    omega=np.asarray(analysis['omega'].data,dtype=complex)
    intens=np.asarray(analysis['IR_intensity'].data,dtype=float)
    signed=np.where(abs(omega.imag)>1e-6,-abs(omega.imag),omega.real)
    table(out/'modes.csv',(dict(mode=i+1,frequency_cm1=float(f),intensity_km_mol=float(v)) for i,(f,v) in enumerate(zip(signed,intens,strict=True))))
    mask=signed>0
    x,y=broaden(signed[mask]*a.scale,intens[mask],0,4000,a.fwhm)
    spectrum_csv(out/'ir_spectrum.csv',x,y,'wavenumber_cm1')
    dump(out/'metadata.json',dict(method=a.method,energy_hartree=float(energy),
        scale=a.scale, imaginary_modes=int(np.sum(signed < -20)),
        warning='Imaginary modes must be investigated; gas-phase harmonic IR is not a crystal spectrum'))

if __name__ == '__main__': main()

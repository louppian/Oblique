"""Outline main analysis. Run directly; one CSV and PNG."""
from _shared import CONFIG, run

def main():
    run('01_noise_concentration',(('noise',('exp01_image_noise_convergence','exp02_text_noise_convergence')),),CONFIG,include_magnitude=False)

if __name__=='__main__':main()

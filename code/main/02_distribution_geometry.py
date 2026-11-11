"""Outline main analysis. Run directly; one CSV and PNG."""
from _shared import CONFIG, run

def main():
    run('02_distribution_geometry',(('geometry',('_section_geometry',)),),CONFIG,include_magnitude=False)

if __name__=='__main__':main()

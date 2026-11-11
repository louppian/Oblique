"""Outline main analysis. Run directly; one CSV and PNG."""
from _shared import CONFIG, run

def main():
    run('04_reference_controls',(('controls',('ensure_corruptions','_section_a7')),),CONFIG,include_magnitude=False)

if __name__=='__main__':main()

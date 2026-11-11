"""Outline main analysis. Run directly; one CSV and PNG."""
from _shared import CONFIG, run

def main():
    run('03_corruption_response',(('corruption',('ensure_geometry','_section_a6')),),CONFIG,include_magnitude=False)

if __name__=='__main__':main()

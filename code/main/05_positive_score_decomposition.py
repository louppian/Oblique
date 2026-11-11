"""Outline main analysis. Run directly; one CSV and PNG."""
from _shared import CONFIG, run

def main():
    run('05_positive_score_decomposition',(('positive_scores',('_section_a9',)),),CONFIG,include_magnitude=True)

if __name__=='__main__':main()

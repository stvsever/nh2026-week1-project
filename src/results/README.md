# Results

This folder holds the group's held-out **p-factor predictions** for the challenge.

## `pfactor_test_predictions.csv`

Written automatically by
[`../jupiter_notebooks/pfactor_multimodal_prediction.ipynb`](../jupiter_notebooks/pfactor_multimodal_prediction.ipynb)
(section 8). It contains one row per participant whose **true `p_factor` is missing (NaN)** in
`participants.tsv`, i.e. exactly the participants the NeuroHackademy organizers ask us to predict:

| column | meaning |
|--------|---------|
| `participant_id` | RBC/PNC participant id |
| `p_factor_predicted` | predicted p-factor from the winning configuration |

The prediction uses the configuration that maximized held-out R-squared in the notebook's tier x algorithm
search (in practice demographics + global structural with a regularized linear or stacking model;
held-out R-squared about 0.08, r about 0.28). Locally only a placeholder participant or two are unlabeled;
when the organizers release the real withheld test set, re-running the notebook fills this file with all of them.

## Submitting

Rename the CSV to your group name if required, commit it, push to GitHub, and open a pull request against the
`results` branch of the challenge template repository. Do not overwrite other groups' files.

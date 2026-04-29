# Cross-Country Layered Finalist Summary

- Best overall finalist by mean $F_{1,\mathrm{wc}}$: `Sanity -> Z-score` (0.6577, $G_{\mathrm{wc}}=0.8776$)
- Second-best finalist: `Sanity -> IF` (0.5860, $G_{\mathrm{wc}}=0.8430$)
- Highest-recall finalist: `Sanity -> Z-score (>=5) -> IF` (mean recall 0.8897, std 0.0467)
- Scope-mh $F_{1,\mathrm{wc}}$ wins: `{"Sanity -> IF": 20, "Sanity -> Z-score": 48, "Sanity -> Z-score (>=10) -> IF": 0, "Sanity -> Z-score (>=5) -> IF": 0}`
- Scope-mh $G_{\mathrm{wc}}$ wins: `{"Sanity -> IF": 30, "Sanity -> Z-score": 35, "Sanity -> Z-score (>=10) -> IF": 1, "Sanity -> Z-score (>=5) -> IF": 2}`
- Scope-mh recall wins: `{"Sanity -> IF": 8, "Sanity -> Z-score (>=10) -> IF": 1, "Sanity -> Z-score (>=5) -> IF": 59}`

The aggregate comparison therefore still favors `Sanity -> Z-score` on the balanced criteria, whereas the recall-first comparison favors `Sanity -> Z-score (>=5) -> IF`.

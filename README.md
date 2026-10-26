# policyengine-emtr-frontend

Interactive EMTR explorer for USC PPD 661 (Public Finance and Poverty Programs).
Live: https://mattunrath.github.io/policyengine-emtr-frontend/

Static site: `index.html` plots pre-computed PolicyEngine US results from `data/<year>/`.
Regenerate with `generate_data.py` (see its docstring). Tax years 2023–2025, 50 states,
0–4 children, Medicaid on/off; earnings $0–$100k in $2k steps.

Net income = earnings + SNAP + TANF + Medicaid value − federal income tax − state income tax,
where the tax lines are already net of refundable credits (EITC, CTC). Payroll taxes excluded.
Maintenance notes live in the course's Overleaf project (`EMTR_WEBSITE_MAINTENANCE.md`).

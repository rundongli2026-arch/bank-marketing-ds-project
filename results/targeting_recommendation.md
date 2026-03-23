# Targeting Recommendation

## Assumptions Used
- Cost per contact: 2.00
- Revenue per conversion: 120.00
- Test-set baseline conversion: 0.474

## Recommended Strategies
1. Budget-limited campaign: prioritize top 10% scored customers (conversion=0.900, lift=1.90x).
2. Profit-oriented strategy under current assumptions: threshold=0.450 (contacted share=0.429, expected net gain=104886.00, ROI proxy=43.81).
3. If recall is prioritized while keeping precision above baseline: use threshold=0.40 (precision=0.695, recall=0.741, contacted share=0.504).

## Notes
- These profit figures are simulation outputs under stated assumptions, not observed production outcomes.
- Profit recommendation is constrained to contacted share <= 0.50 to reflect finite campaign capacity.
- Final threshold should be tuned jointly with outreach budget, capacity, and compliance constraints.
# Overview about the dataset preprocessing process

| Dataset Collection (folder names)      |Status | Notebook       | Comments                                                                                     |
| :------------------------------------- |:-----:|:---------------|:---------------------------------------------------------------------------------------------|
| ATLAS Higgs Boson Challenge            |   x   | [🗎][ATLAS]    | Classification dataset; time component arbitrary                                             |
| Community-NAB                          |   ✓   | [🗎][NAB]      |                                                                                              |
| IOPS AI Challenge                      |   ✓   | [🗎][IOPS]     |                                                                                              |
| KDD Robot Execution Failures           |   x   |                | Only very short sequences and annotations are per sequence instead of per point!             |
| MIT-BIH Arrhythmia DB                  |   ✓   | [🗎][mitdb]    | Complex generation of anomaly-windows to label datasets.                                     |
| MIT-BIH Long-Term ECG Database         |   ✓   | [🗎][ltdb]     | See _MIT-BIH Arrhythmia DB_ for preprocessing explanation.                                   |
| MIT-BIH Supraventricular Arrhythmia DB |   ✓   | [🗎][svdb]     | See _MIT-BIH Arrhythmia DB_ for preprocessing explanation.                                   |
| NASA Spacecraft Telemetry Data         |   ✓   | [🗎][NASA]     | SMAP and MSL datasets                                                                        |
| Series2Graph                           |  tbd  |                | **No labels ATM!**                                                                           |
| Server Machine Dataset                 |   ✓   | [🗎][SMD]      |                                                                                              |
| TSBitmap                               |   x   |                | **No labels!**                                                                               |
| UCI ML Repository / 3W                 |   x   | [🗎][3W]       | Hard to transform into TS AD task.                                                           |
| UCI ML Repository / CalIt2             |   ✓   | [🗎][CalIt2]   |                                                                                              |
| UCI ML Repository / Condition monitoring|   x   | [🗎][Cond]     | Whole-sequence annotations; multiple condition annotations per sequence!                    |
| UCI ML Repository / Daphnet            |   ✓   | [🗎][Daph]     |                                                                                              |
| UCI ML Repository / Dodgers            |   ✓   | [🗎][Dodgers]  | Missing values are marked as anomalies as well.                                              |
| UCI ML Repository / HEPMASS            |   x   |                | Classification dataset; time component arbitrary                                             |
| UCI ML Repository / Kitsune Network Attack|   ✓   | [🗎][Kitsune] | Very large datasets; distance between points (network packets) unclear                     |
| UCI ML Repository / Metro              |   ✓   | [🗎][Metro]    |                                                                                              |
| UCI ML Repository / OPPORTUNITY        |   ✓   | [🗎][OPP]      | To-Lie is regarded as anomaly. A lot of missing values!                                      |
| UCI ML Repository / Occupancy Detection|   ✓   | [🗎][Occu]     |                                                                                              |
| UCI ML Repository / URLReputation      |   x   |                | No real time series; labels are per item, but no way to follow an item over the time period. |
| Webscope-S5                            |   ✓   | [🗎][Yahoo]    |                                                                                              |
| credit-card-fraud                      |   x   |                | Timestamps are not equi-distant.                                                             |
| genesis-demonstrator                   |   ✓   | [🗎][gen]      | A single dataset                                                                             |
| GHL                                    |   ✓   | [🗎][ghl]      |                                                                                              |
| SSA                                    |   ✓   | [🗎][ssa]      | Annotation source unclear, brittle datasets.                                                 |
| Keogh                                  |   ✓   | [🗎][keogh]    | Collection of datasets from Eammon Keogh                                                     |
| MGAB                                   |   ✓   | [🗎][mgab]     |                                                                                              |
| KDD-TSAD-contest                       |   ✓   | [🗎][kdd-tsad] |                                                                                              |
| SWaT                                   |   ✓   | [🗎][swat]     |                                                                                              |
| WADI                                   |   ✓   | [🗎][wadi]     |                                                                                              |
| TSB-UAD                                |   ✓   | [🗎][TSB-UDA]  | benchmark datasets are already included in our other collections                             |

## TODO

Check against datasets in [John's benchmark framework](https://github.com/johnpaparrizos/AnomalyDetection/tree/master/benchmark/dataset):

- ECG (source are mitdb, ltdb, and svdb, label source unknown)
- GHL ✓
- NAB ✓
- SMAP ✓
- SMD ✓
- SSA ✓
- YAHOO ✓

[gen]: ./Genesis%20Demonstrator.ipynb
[mitdb]: ./MIT-BIH%20Arrhythmia%20Database.ipynb
[ltdb]: ./MIT-BIH%20Long-Term%20ECG%20Database.ipynb
[svdb]: ./MIT-BIH%20Supraventricular%20Arrhythmia%20DB.ipynb
[NAB]: ./NAB.ipynb
[NASA]: ./NASA%20Spacecraft%20Telemtry.ipynb
[SMD]: ./Server%20Machine%20Dataset.ipynb
[Yahoo]: ./YahooWebscopeS5.ipynb
[IOPS]: ./IOPS%20AI%20Challenge.ipynb
[ATLAS]: ./ATLAS%20Higgs%20Boson%20Challenge.ipynb
[3W]: ./UCI-3W.ipynb
[CalIt2]: ./UCI-CalI2.ipynb
[Cond]: ./UCI-Condition%20Monitoring.ipynb
[daph]: ./UCI-Daphnet.ipynb
[Dodgers]: ./UCI-Dodgers.ipynb
[Kitsune]: ./UCI-Kitsune.ipynb
[Metro]: ./UCI-Metro.ipynb
[OPP]: ./UCI-Opportunity.ipynb
[Occu]: ./UCI-Occupancy.ipynb
[ghl]: ./GHL.ipynb
[ssa]: ./SSA.ipynb
[keogh]: ./Keogh.ipynb
[mgab]: ./MGAB.ipynb
[kdd-tsad]: ./KDD-TSAD.ipynb
[swat]: ./SWaT.ipynb
[wadi]: ./WADI.ipynb
[TSB-UDA]: ./TSB-UAD.ipynb

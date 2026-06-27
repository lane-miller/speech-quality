# Findings #

## Compiled Data ##

**Selected results per degradation (Spearman > 0.6)**
*Note variances are relative (per degradation) and range normalized due to scale differences, and rated L,M,H for low medium or high on range 0-1*
| degradation | metric | spearman_rho_vs_severity | shape | response_range | variance | rho_vs_dnsmos_ovr | rho_vs_nisqa |
|---|---|---|---|---|---|---|---|
| clipping | nisqa | 0.729 | sat mild | severe | M | 0.57 | 1.0 | 
| clipping | clipping rate | -0.948 | sat mild | severe | L | -0.34 | -0.75 |
| clipping | crest factor | 0.878 | monotonic | full | M | 0.34 | 0.73 | 
| lowpass | nisqa | 0.784 |  monotonic | full | M | 0.65 | 1.0 | 
| lowpass | spectral flatness | 0.978 | sat severe | mild | L | 0.46 | 0.80 | 
| lowpass | hf energy ratio | 0.920 | sat severe | mild | L | 0.42 | 0.74 |
| reverb | dnsmos ovr | -0.735 | sat severe | mild | M | 1.0 | 0.80 |
| reverb | nisqa | -0.748 | monotonic | full | M | 0.80 | 1.0 |
| reverb | wada snr | -0.676 | monotonic | full | M | 0.60 | 0.66 |
| reverb | c50 | -1.00 | monotonic | full | L | 0.74 | 0.75 |
| noise pink | dnsmos ovr | 0.878 | monotonic | full | M | 1.0 | 0.89 |
| noise pink | nisqa | 0.930 | sat mild | mod - severe | M | 0.89 | 1.0 | 
| noise pink | wada snr | 0.836 | sat mild | mod - severe | L | 0.74 | 0.82 |
| noise babble | dnsmos ovr | 0.915 | sat severe | mild - mod | M | 1.0 | 0.91 |
| noise babble | nisqa | 0.930 | monotonic | full | M | 0.91 | 1.0 |
| noise babble | wada snr | 0.879 | sat mild | mod - severe | L | 0.82 | 0.87 |
| noise babble | pitch conf | 0.623 | sat mild | mod - severe | M | 0.65 | 0.66 |
| noise tonal lf | dnsmos ovr | 0.821 | sat mild | mod - severe | M | 1.0 | 0.65 |
| noise tonal lf | nisqa | 0.632 | sat mild | mod - severe | H | 0.65 | 1.0 |
| noise tonal lf | wada snr | 0.842 | sat mild | mod - severe | L | 0.72 | 0.65 |
| noise tonal lf | pitch conf | 0.683 | sat mild | mod - severe | M | 0.63 | 0.48 |
| noise tonal hf | dnsmos ovr | 0.896 | monotonic | full | M | 1.0 | 0.80 |
| noise tonal hf | nisqa | 0.826 | sat mild | mod - severe | M | 0.80 | 1.0 |
| noise tonal hf | wada snr | 0.843 | sat mild | mod - severe | L | 0.76 | 0.76 |
| noise tonal hf | hf energy ratio | -0.702 | sat mild | severe | M | -0.65 | -0.49 |

**Spearman correlation between DNSMOS and NISQA**
| degradation | dnsmos_nisqa_agreement |
|---|---|
| clipping | 0.566 |
| codec | 0.0418 |
| lowpass | 0.648 |
| reverb | 0.80 |
| noise pink | 0.889 |
| noise babble | 0.914 |
| noise tonal lf | 0.649 |
| noise tonal hf | 0.798 |



## Discussion ##

**Clipping**
- DNSMOS largely insensitive to clipping (ρ≤0.39); NISQA substantially more sensitive (ρ=0.73)
- Weak DNSMOS/NISQA agreement under clipping (ρ=0.57) — likely attributable to DNSMOS training on additive noise conditions
- clipping_rate best physical tracker (ρ=-0.95) but weak perceptual correlation (rho_vs_nisqa=-0.75, rho_vs_dnsmos_ovr=-0.34)
- crest_factor monotonic full-range response (ρ=0.88); moderate NISQA correlation (0.73) — lightweight perceptual proxy for clipping severity per NISQA
- No engineered metric tracks DNSMOS OVR meaningfully under clipping

**Codec**
- No metric reliably tracks codec bitrate across the full sweep (max ρ=0.42)
- NISQA shows localized response at lowest bitrates (6–16 kbps); insensitive above ~16 kbps
- Codec artifacts largely undetected by all engineered metrics at tested severity levels

**Lowpass**
- NISQA most sensitive to lowpass (ρ=0.784, monotonic) but DNSMOS/NISQA agreement moderate (ρ=0.648) — neural metrics partially disagree
- spectral_flatness and hf_energy_ratio are excellent physical trackers (ρ=0.978, 0.920) but saturate at severe degradation — blind to worst-case bandwidth limiting
- No engineered metric reliably tracks perceived quality across the full lowpass range
- Lowpass is a degradation where physical measurement (what frequencies are present) decouples from perceptual impact at severe levels

**Reverb**
- C50 correlates meaningfully with both neural MOS metrics (ρ≈0.75) — strongest engineered-to-perceptual link in the study; note C50 is RIR-derived so T60 tracking is expected by construction
- DNSMOS saturates at severe T60; NISQA monotonic across full range — NISQA preferred for heavy reverberation; neural agreement moderate-strong (ρ=0.80)
- WADA-SNR tracks reverb despite being a noise estimator (ρ=-0.676) — unexpected cross-domain sensitivity

**Noise Pink**
- Strong neural metric agreement under pink noise (ρ=0.889) — DNSMOS and NISQA largely interchangeable for this degradation
- WADA-SNR is a meaningful perceptual proxy (ρ_vs_nisqa=0.82) but saturates above ~25 dB SNR — useful only in moderate-to-severe noise conditions
- DNSMOS OVR tracks monotonically across the full SNR range while NISQA saturates at mild degradation — DNSMOS more reliable at low noise levels

**Noise Babble**
- Strong neural agreement under babble (ρ=0.914)
- WADA-SNR strong physical and perceptual tracker (ρ=0.879, rho_vs_nisqa=0.87) — most reliable engineered metric for babble across moderate-to-severe conditions
- pitch_confidence marginally qualifies (ρ=0.623) but never recovers to baseline even at 40 dB SNR — babble persistently disrupts voicing detection even at mild levels

**Noise Tonal LF**
- Moderate DNSMOS/NISQA disagreement under LF tonal noise (ρ=0.649) — hum disrupts DNSMOS BAK strongly (ρ=0.876) but NISQA less so (ρ=0.632), suggesting DNSMOS is more sensitive to background tonal interference
- WADA-SNR strongest engineered tracker (ρ=0.842) with meaningful perceptual correlation (rho_vs_dnsmos_ovr=0.72, rho_vs_nisqa=0.65)
- pitch_confidence tracks severity (ρ=0.683) but fails to correlate perceptually (rho_vs_nisqa=0.48) — LF hum disrupts voicing detection without proportionally reducing perceived quality

**Noise Tonal HF**
- Better DNSMOS/NISQA agreement (ρ=0.798) than LF tonal (ρ=0.649) — neural metrics converge more under HF tonal noise
- hf_energy_ratio physically detects HF tonal presence (ρ=-0.702) but is perceptually decoupled (rho_vs_nisqa=-0.49) — spectral energy evidence of noise does not map proportionally to perceived degradation
- WADA-SNR most balanced engineered tracker: meaningful severity correlation (ρ=0.843) and strongest perceptual alignment (rho_vs_dnsmos_ovr=0.76, rho_vs_nisqa=0.76) — more perceptually predictive than in the LF tonal condition

**Noise Impulsive**
- No metric reliably tracks click rate across the tested severity range at moderate SNR levels (10+ dB)


# AdaptiveAttractorForSequences
## Sequential activation in adaptive hippocampal attractor networks

### Required python packages top reproduce figures below:
> python 3.8 and later versions\
> brainpy\
> add the packages needed to reproduce the results here...

### Fig1: An adaptive attractor network for the place cell population
- [ ] Figure 1.1: model schematic

- [x] Figure 1.2: place field

### Fig2: Delayed spatial encoding and delay compensation by adaptation
- [ ] Figure 2.1: localized bump-like network activity (1-dimensional demonstration)

- [ ] Figure 2.2：smooth tracking of the external location input $I_{ext}$

- [x] Figure 2.3: delay distance v.s. moving speed

- [x] Figure 2.4: delay distance v.s. adatation strength, we should observe that delay distance decrease as the adaptation strength increases

- [x] Figure 2.5: fixing $m=\tau/tau_v$, plot delay distance as a function of moving speed

### Fig3: Online sequential activation resembles theta sweeps
- [x] Figure 3.1: forward theta sweeps

- [ ] Figure 3.2: phase precession (unimodal cell)

- [x] Figure 3.3: bi-directional theta sweeps

- [ ] Figure 3.4: phase precession interleaved with phase procession (bimodal cell)

### Fig4: Offline sequential activation resemble hippocampal replay dynamics
- [x] Figure 4.1: demonstration of diffusion and super-diffusion in 2D space

- [x] Figure 4.2: histogram plot of probability v.s. step sizes for diffusion and super-diffusion

- [x] Figure 4.3: Phase diagram

- [ ] Figure 4.4: Levy exponent v.s. adaptation strength; Levy exponent v.s. noise strength

### Fig5: Anti-correlation between neural activity and movement trajectory
- [x] Figure 5: already done

---

## @Tianhao & Xingsi: please modify each of the figures according to the hints below. Also please use the same colormap as much as possible during plot.
#### Figure1.2: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig1_2_placefield.png" width=40% height=40%>
</p>
<em>Hint: Xaxis: switch to location. Yaxis: switch to cell index. Add noise, no smooth, no periodic boundary (cut off the two sides). Add colorbar. See Fig1c in [https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-23765-x/MediaObjects/41467_2021_23765_MOESM1_ESM.pdf] for an example.</em>

#### Figure2.3: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig2_3.png" width=40% height=40%>
</p>
<em>Hint: Xaxis: Moving speed (rads/s), values are too small, need to match the real moving speed if it is on the linear track. Yaxis: Lag distace （rads）. Change to scatter plot with linked lines between scatters. Add noise during simulation and do shadow plot.</em>

#### Figure2.4: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig2_4.png" width=40% height=40%>
</p>
<em>Hint: Xaxis: Adaptation strength (m). Yaxis: Lag distance（rads）. Change to scatter plot with linked lines between scatters. Add noise during simulation and do shadow plot.</em>

#### Figure2.5: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig2_5.png" width=40% height=40%>
</p>
<em>Hint: Xaxis: Moving speed (rads/s), values are too small, need to match the real moving speed if it is on the linear track. Yaxis: Lag distance（rads）. Change to scatter plot. Add noise during simulation and do shadow plot.</em>

#### Figure3.1: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig3_1.png" width=40% height=40%>
</p>
<em>Hint: Xaxis: Time ms to s. Mark the theta rhythm boundary in a clearer way. Time starts from 0s and ends up at 1s. Yaxis: Change encoded position to decoded position. Add the caption for the colorbar.</em>

#### Figure3.3: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig3_3.png" width=40% height=40%>
</p>
<em>Hint: refer to Fig3.1.</em>


#### Figure4.1: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_1.png" width=40% height=40%>
</p>
<em>Hint: refer to Fig4.1.</em>

#### Figure4.2: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_2.png" width=40% height=40%>
</p>
<em>Hint: refer to Fig4.2.</em>

#### Figure4.3: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_3.png" width=40% height=40%>
</p>
<em>Hint: refer to Fig4.3.</em>


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
- [x] Figure 2.1: localized bump-like network activity (1-dimensional demonstration)

- [x] Figure 2.2：smooth tracking of the external location input $I_{ext}$

- [x] Figure 2.3: delay distance v.s. moving speed

- [x] Figure 2.4: delay distance v.s. adatation strength, we should observe that delay distance decrease as the adaptation strength increases

### Fig3: Online sequential activation resembles theta sweeps
- [ ] Figure 3.1: illustration of the interplay between intrinsic dynamics and external snesory-motion information
- [x] Figure 3.2: forward theta sweeps
- [x] Figure 3.3: bi-directional theta sweeps
- [x] Figure 3.4: quantitative measurement of forward sweeps and bidirection sweeps.

- [x] Figure SIxxx (move to SI): phase precession (unimodal cell)
- [x] Figure SIxxx (move to SI): phase precession interleaved with phase procession (bimodal cell)

### Fig4: Offline sequential activation resemble hippocampal replay dynamics
- [x] Figure 4.1: demonstration of diffusion and super-diffusion in 2D space

- [x] Figure 4.2: histogram plot of probability v.s. step sizes for diffusion and super-diffusion

- [x] Figure 4.3: Phase diagram

- [x] Figure 4.4: Levy exponent v.s. adaptation strength; Levy exponent v.s. noise strength

### Fig5: Anti-correlation between neural activity and movement trajectory
- [x] Figure 5: already done

---

## @Tianhao & Xingsi: please modify each of the figures according to the hints below. Also please use the same colormap as much as possible during plot.
#### Figure1: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig1.pdf" width=40% height=40%>
</p>
<em>Hint: Xaxis: switch to location. Yaxis: switch to cell index. Add noise, no smooth, no periodic boundary (cut off the two sides). Add colorbar. See Fig1c in [https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-23765-x/MediaObjects/41467_2021_23765_MOESM1_ESM.pdf] for an example.</em>
#### Figure2: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig2.pdf" width=40% height=40%>
</p>
#### Figure3: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/fig3.png" width=40% height=40%>
</p>
#### Figure4.1: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_1.png" width=40% height=40%>
</p>
<em>Hint: @Xing si: to show the difference of Levy flight and diffusion, you have to put the scale on x and y axis. Colorbar font size is larger than font size under x axis. Make them to be consistent. </em>

#### Figure4.2: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_2.png" width=40% height=40%>
</p>
<em>Hint: Make sure the color map used here is consistent with other plots. Chage the font size of x and y labels.</em>

#### Figure4.3: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_3.png" width=40% height=40%>
</p>
<em>Hint: Remove the image smoothness.</em>

#### Figure4.4: 
<p float="center">
<img src="https://github.com/ZilongJi/AdaptiveAttractorForSequences/blob/main/Figures/Fig4_4.png" width=40% height=40%>
</p>
<em>Hint: Remove the image smoothness.</em>



# Robot Vision and Navigation

- lowercase letter: scalar
- lowercase bold letter: vector
- uppercase bold letter: matrix
- x states / measurement function parameters
- y input
- z output
- $\bold{H}$ derivative/measurement/design matrix $\frac{\partial h_i}{\partial x_i}$ / homogenous y 
- $\bold{h}$ measurement function.
- $\hat{x}$ estimated value
- $\hat{x}^-$ predicted value
- $\tilde{x}$ value with error
- $x^+$ posteriori
- $x'$ best guess of the true value of the states.



- $p^s_a$ pseudo range, distance from satellite to antenna + clock offset in meters.
- $\dot{p}^s_a$ pseudo range rate, rate from satellite to user antenna + clock drift in meters.
- $\Delta f^{s}_{ca,a}$ doppler shift Hz = -pseudo range rate * carrier frequency / speed of light.
- $\Upphi^s_a$ carrier phase / accumulated delta range (ADR).
- $\hat{r}^e_{ea}$ estimated (^) cartesian position (r) of user antenna (a) with respect to and resolved about origin of ECEF frame (e)
- $\tilde{p}^s_{a,C}$ measured pseudo-ranges from satellites (s) to user antenna (a), corrections applied (C)
- $\delta\hat{p}^a_c$ estimated (^) range error at the user antenna (a) due to receiver clock offset (c)



- bearing: bearing: angle of the horizontal plane between two line-of-sight to an object and known direction (magnetic north).
- 



### Positioning

- Realtime or Postprocessed
- Fixed or Movable?
- Self or Remote positioning

Navigation is Realtime, Movable and Self positioning.

Methods:

- Position fixing - use environmental feature / landmark (aid to navigation (AtoN)) to determine position.
  - Proximity: location of user is set to location of the landmark, multiple landmarks then average them (for close range application such as bluetooth). 
  - Advanced proximity: containment intersection where intersection of landmark zones is used to estimate user position.
  - Line-of-position (LOP) where intersection of circle d efined by radius reaching the user is used (In 3D, it is called surface-of-position (SOP)). The range can be determine by signal transmission, in the case of two-way transmission, most synchronization error can be cancelled out; in the case of one-way transmission such as GNSS, the receiver clock offset is treated as an additional unknown in the position solution, this can be resolved using an additional satellite or reference receiver with known location. When landmark is an environmental feature, a active sensor is needed to measure the round-trip-time (RRT) (such as radar, sonar or laser).
  - 
- Dead reckoning - initial position + measure distance & direction travelled
  - Inertial navigation system (INS) = Inertial measurement unit (IMS) + navigation processor (initial position, velocity & attitude & gravity model) -> current position, velocity & attitude.

### Satellite Position & Velocity

The satellite motion is described by a two-body kaplerian model (baseline model)

<img src="COMP0130 Robot Vision and Navigation.assets/image-20220124172508822.png" alt="image-20220124172508822" style="zoom: 67%;" />

- ascending node: orbit crosses Earth's equatorial plane while the satellite moving in positive z-direction.
- descending node: same as ascending node but moving in negative z-direction.



### Single Epoch GNSS Positioning

$r_{a s}^{2}=x_{a s}^{e^{2}}+y_{a s}^{e^{2}}+z_{a s}^{e^{2}}$, by pythagorus' theorem twice = user to satellite distance.

where:
$$
\begin{array}{ll}
\mathbf{r}_{a s}^{e}=\mathbf{r}_{e s}^{e}-\mathbf{r}_{e a}^{e} & y_{a s}^{e}=y_{e s}^{e}-y_{e a}^{e} \\
x_{a s}^{e}=x_{e s}^{e}-x_{e a}^{e} & z_{a s}^{e}=z_{e s}^{e}-z_{e a}^{e}
\end{array}
$$
Together we have: $r_{a s}^{2}=\left(x_{e s}^{e}-x_{e a}^{e}\right)^{2}+\left(y_{e s}^{e}-y_{e a}^{e}\right)^{2}+\left(z_{e s}^{e}-z_{e a}^{e}\right)^{2}$.

We add the clock offset: $\rho_{a, C}^{s}=r_{a s}+\delta \rho_{c}^{a}$ to get the equation for the user distance.

But we only know the satellite positions and do not know the user positions (3 unknowns) and range error (1 unknown) due to receiver clock.

So we need 4 satellite to solve for these 4 unknowns. 





## Stanford Course

- Orbits rotate planet in ellipses
- Time taken to travel some distance d along that ellipses is the area of d to the planet (the closer to the planet, the faster you move).
- Period square is proportional to the semi-major axis cubed (?)

### Satellite Clocks

- Einstein's general relativity: less gravity means time move slower
  - Satellite in the sky has less gravity than on earth: +45us/day
- Einstein's special relativity: move faster means time move faster
  - Satellite move at about 3km/s: -7us/day

Note the satellite travel around the orbit, which has different height (different gravity) and different speed, so need to calculate these in real-time. (eccentric anomaly )

Net result: clock is programmed so that they move slower, about +38us/day.

### Orbits

We have Medium Earth Orbit (MEO), Low Earth Orbit (LEO) and GEO, MEO is for satellite navigation, LEO is closer to earth, so that it can take pictures / spy etc... GEO is fixed relative to earth's surface (?)

### Messages

Bits/sec is low, because signal is quite weak when it reach the ground, so data compression is needed and encode everything into few bits.

<img src="COMP0130 Robot Vision and Navigation.assets/image-20220113182959609.png" alt="image-20220113182959609" style="zoom:33%;" />

It takes 6 seconds to send each segment, so 30 seconds in total for all 6 segments above.

### Keplerian Elements

- describe the shape of the orbit
  - Diameter $2a$, called major axis; $a$ is the semi-major axis, 
- describe the rotation of orbit
  - $i$ the tilt angle of the orbit relative to equatorial plane. 
  - $\Omega$ the angle (starts from Vernal Equinox) where the orbit intersect with the equatorial plane. (right ascension of the ascending node)
  - $\omega$ the angle from the ascending node to to the Perigee.
  - $v$ the angle starts from the Perigee to denote the location of the satellite.

### Signals

<img src="COMP0130 Robot Vision and Navigation.assets/image-20220113200027072.png" alt="image-20220113200027072" style="zoom: 50%;" />

Different frequencies are used by different satellite systems and various purpose (civil or not...)

A pseudo-random signal has special structure, they have sent with a replica and they have good correlation properties (if there is a slight shift, they will have weak correlation), the receiver can check the correlation of two received signals, if they have strong correlation, then we will get correlation peak which suggests we are synchronized

### Capabilities

- Finds location (error within 5m)
- Correct user time
- Velocity estimation, based on doppler shift, the satellite send a signal to the user, but the user may be moving, so when user receives the signal, the weaker it is, the further away the user travel since the satellite send the signal, so we can find out the user velocity. (error within 0.2m/s)

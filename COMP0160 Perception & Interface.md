# COMP0160 Perception & Interface

## Lectures

### Lecture 1

Perception refers to how we experience the world through our five senses. It helps us to gain understanding and interpretation of our environment, and help us to plan our action in response to this sensory information. It includes three stages: **organize**, **identification** and **interpretation**.

#### Perceptual Process

- Stimulus
  - Environmental / Attended / Receptor
  - When detected, it calls one of our 5 senses.
- Physiological Process (Neural Activation)
  - Transduction, Transmission, Processing
  - Electrical signals are generated in response to the stimulus and transmitted to the brain.
- Experience & Action
  - Goal of perception, recognition and reaction, categorizing and acting on the stimulus.
- Knowledge
  - Prior knowledge about the world, influences response that applied.;m

The process can be studied through physiological approach and psychophysical approach

- Physiological: relation of the physiological process and the resultant action or the relation between stimulus and the physiological process. (nervous system / chemical process)
- Psychophysical: relation of the stimulus, physics, light / sound waves and resultant action. (experiments)

#### Basics of Psychophysics

The main idea is to change an external property of the stimulus (visual, taste, touch, auditory) and measure the behavioral parameter (accuracy, reaction time, sensory threshold). We can use an objective scale of measurement because the result is quantitative.

Participants can provide the information such as: description, search, differences, detect, describe, recognize.

Different thresholds:

- Absolute threshold: slowly increase until the user say yes.
- Difference threshold: slowly increase the difference until the user can sense the difference.

To find the threshold:

- Method of adjustment: participant change the threshold until they show positive.
- Method of limit: participant say yes / no when experimenter change the threshold ascending / descending.
  - error of habituation: observer may become accustomed to report the same value even when threshold is passed.
  - error of expectation: observer may anticipate the stimulus is about to become change state and make premature judgement.
- Method of constant stimuli: same as limit but change the threshold randomly, this can avoid the errors above. 

The graph plotted external parameter against participant result is called **psychometric function**. From the graph we can figure out the threshold. But there are always **noise** in experiments.  Noise can come from **neural activity**, **stimulus (physical)**, and **attention**.

To avoid this, we can use **adaptive staircase** techniques, where we presents lower the threshold if participant cannot detect and raise if can be detected (we starts with big steps and slowly decrease the steps, to get a more accurate response), in this case, we spend more time around the threshold and can be good for fitting psychometric function, but SDT can do better:

#### Signal Detection Theory (SDT)

We can use SDT to analysis the participant, by identifying hit/false alarm/miss/correct rejection we can figure out whether the participant is more willing to say yes/no (low/high criterion), and we aimed to find out although the participants have different criteria for responding to the stimuli, but is their underlying sensitivity to the threshold the same?

We can measure the effect of criterion using two distributions:

<img src="COMP0160 Perception & Interface.assets/image-20220121113019809.png" alt="image-20220121113019809" style="zoom:50%;" />

The less overlaps between two distributions means less misses and false alarms. The overlaps are defined by the mean of distribution and the standard derivation, calculated as $d'$:
$$
d'=\frac{\left(mean_{Signal+Noise}-mean_{Noise}\right)}{std_{Noise}}
$$
However, the distributions are not usually known, therefore $d'$ is usually calculated from a table of data $z$:
$$
\begin{align*}
d'&=\text{hit rate} - \text{false alarm rate}\\
  &=P(yes|SN) - P(yes|N)\\
  &=z(yes|SN) - z(yes|N)
\end{align*}
$$
where $S$ is signal and $N$ is noise. Usually, 0 is chance level performance, 1 is moderate and 4.65 is optimal (hit rate 0.99 and false alarm 0.1). Now if we plot the ROC curve, we can further explore the differences between sensitivity and criterion.

#### Physiological Process

- Occipital lobe for receiving visual information 
- Temporal lobe for receiving auditory information 
- Parietal lobe for receiving tactile information 
- Frontal lobe for receiving information from all the senses and is involved In the coordination of information received through multiple senses.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Cerebrum_lobes.png/1920px-Cerebrum_lobes.png" alt="img" style="zoom: 25%;" />

Neurons consists of a **cell body** appended with **axon** and **dendrites**; dendrites are inputs and can have thousands of them; and axon is the output and has only one; neurons communicate via transduction or electric flow. Together they form sense receptors that handles different environmental information from our five senses.

### Lecture 3

#### Structure of Ear

1. Outer ear
   1. **pinna 表面** act as a funnel, filter and help localisation of sound.
   2. **auditory canal 耳道** boost sensitivity in the range 1000hz-5000hz.
2. Middle ear
   1. **eardrum 鼓膜** consists of three bones, change pressure variation
   2. **eustachian tube 咽鼓管** equalize pressure between middle ear and oral canvity
3. Inner ear
   1. **cochlea 耳蜗** bony structure filled with fluid, transform pressure variation to neural impluses; one of the bones (stapes) from eardrum attach to oval window of cochlea.
      - different parts have different **stiffness 刚度** that react to different frequencies.
      - the conversion from wave to electrical signals is done via **corti**, it has hair on its surface that depolarize when it is vibrated; causing the neurontransmitter to release; and leads to potentials transferred to higher auditory area.
        - cochlear $\rightarrow$ Auditory Brainstem $\rightarrow$ Auditory Forebrain

#### Sound Perception

- loudness: amplitude.
- sensitivity range: 20-20k
- pitch: cycle frequency
- tone height: increasing pitch and fundamental frequency at the same time.
- fundamental frequency: the lowest frequency of N periodic waveform.

### Localization

- Visual: objects are imaged on different places of the retina
- Sound: vibrations are spread across the cochlea
  - folds





## Coursework 1

- Point Spread Function (PSF): This describe the response of an imaging system to a point source. (a system's impulse response)

- acuity 敏锐度

- contrast sensitivity 对比敏感度

- aperture 光圈

- retina 视网膜

  

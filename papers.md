# Papers

### StyLit: Illumination-Guided Example-Based Stylization of RevColor3D Renderings

> https://dcgi.fel.cvut.cz/home/sykorad/Fiser16-SIG.pdf

#### Overall

Use light propagation instead of normal & colors to guide.

1. Artist pay attention to color as much as lightning.	

   1. same color can painted differently due to lighting
   2. color can be deviated to enhance effect of lighting
   3. normal is not enough for advance illumination effects, such as shadow and glossy reflections.

   Solution:

   1. Compute a simple light propagation:

      <img src="Untitled 1.assets/image-20220213161316136.png" alt="image-20220213161316136" style="zoom: 33%;" />

   2. Then ask different artists to provide painting (illumination dependent stylization):

      <img src="Untitled 1.assets/image-20220213161418835.png" alt="image-20220213161418835" style="zoom:33%;" />

2. Previous method distort high level feature such as brush size and reuse a small subset of patches, producing distinct wash-out effects / homogenous regions

   Solution:

   - encourage uniform usage of source patches while controlling the overall error to avoid enforcing use of patches that cause disturbing visual artifacts.

#### Related

Convert photo / cg-image to digital painting:

- physical simulation  [Curtis et al. 1997; Haevre et al. 2007]
- procedural techniques [Bousseau et al. 2006; Benard et al. 2010 ´ ]
- advance image filtering [Winnemoller et al. 2012 ¨ ; Lu et al. 2012]
- algorithmic composition of exemplar strokes [Salisbury et al. 1997; Zhao and Zhu 2011]
- Zeng et al. [2009] decomposed the stylized image into a set of meaningful parts for which semantic interpretation is available [Tu et al. 2005], and then modified the stroke placement process to better convey the semantics of individual regions

All these techniques provide impressive results only in some cases.

- Sloan et al. [2001] introduced The Lit Sphere—a generic example-based technique that transfer a painted shaded spheres' pixels to target 3d model using environmental mapping  [Blinn and Newell 1976], ok result but leads to disturbing artifacts
  - extended to handle animation  [Hashimoto et al. 2003; Benard et al. 2013 ´ ] and control the spatial location as well as local orientation of the source textural features in the synthesized image [Wang et al. 2004; Lee et al. 2010].
-  synthesis algorithm has been replaced by a texture optimization technique [Kwatra et al. 2005; Wexler et al. 2007]
  - but also introduce wash-out effect  caused by excessively reusing patches with low frequency content [Newson et al. 2014].
    - many techniques introduced, such as a discrete solver [Han et al. 2006], feature masks [Lefebvre and Hoppe 2006], color histogram matching [Kopf et al. 2007], and bidirectional similarity [Simakov et al. 2008; Wei et al. 2008]. 
    - Kaspar at al. [2015] and Jamriska et ˇ al. [2015] showed that above techniques only works in some particular cases and introduced content-independent solutions that encourage uniform patch usage, but it cannot be used here because it is not actually uniform.
- recently uses computer assisted stylization by  Gatys et al. [2015], It uses a deep neural network trained on object recognition [Simonyan and Zisserman 2014] to create mapping between style feature and target image
  - Although this technique produces impressive results, it learns common visual features from a generic set of natural images instead of stylized features and thus does not fit the task of style transfer for rendered scenes.
  - It also only depends on statistics of color patterns and provide no intuitive means to transfer process, so the style tranfer is unpredictable.
- Diamanti et al. [2015] showed limited examples of additional annotations such as normals and a simple descriptor can be sythesized to form complex material appearance, but cannot capture complex lighting phenomena; they also uses image melding which also prone to wash-out problem described above. 

#### workflow

1. prepare a 3d scene that contains all illuminations needed for subsequent scene (typically, sphere on a table)
2. render the scene using global illumination algorithm  [Kajiya 1986], 
3. print it on a paper in low contrast with special alignment marks
4. artist paint on this paper using any preferred media
5. align artist painting with rendered image using alignment marks.
6. 



### Character Motion Animation Review

> https://crad.ict.ac.cn/EN/abstract/abstract3078.shtml

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220217_1645091715.png" alt="image-20220217095515263" style="zoom: 80%;" />

- Spacetime optimization
  - make use of character motion pattern and optimization techniques, translate physical motion constraints to mathematical constraints
    - degree of freedom is high, computational expensive
    - control character through adjusting loss function and constraint, but not realtime.
    - lots of variables, high non-linearity, hard to control high level parameters such as movement speed.
    - e.g. luxo, locomotion
    - improvements made
      - motion can be divided and simplified
      - reduce dimensionality using relationship between joint motion
      - during interaction of two characters, update one character motion at a time and optimize their constraints together, reducing number of variables.
      - Use covariance matrix to solve large scale non-linear optimization problem.
- kinetic constraint optimization
  - comes from control optimization theory, uses multi-target optimization model to satisfy real-time high level parameter control task. In contrast to spacetime optimization, it optimize motion in a limited period of time, have similar problems.
    - degree of freedom is high, motion difference between each joint is high, computational expensive
    - In partial optimization model, it is a multi-target (character motion, style, balancing, etc...) optimization problem, nontrivial to solve.
    - either reference to character pose / character motion
- low dimensionality modelling
  - reduce number of joints (or replace by spring), extract the overall motion information and estimate the final character motion.
- limited state controller
  - view motion as switching between multiple states, set a proportional-derivative at each joint that controls the character's transition from one state to another.
    - computational efficient, 1000fps
    - hard to design controller
    - hard to generalize, need to manually adjust state machine in different environment
    - derived from machine control theory, character moves like machine animation
  - in early time, hyperparameters are given by experienced researchers.
- data-driven approach
  - physical motion equation calculation driven by character motion animation capture data, ongoing trends
- kinetic filtering
  - filter the right motion from existing database, and calculate the transitions between them, but often not very natural; in contrast to data-driven approach, it also tracks motion capture data in the database, increasing trends
- statistical models
  - use low dimensionality model to model full body motion, focus on how to move to low dimensionality, often estimates based on database

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/02/upgit_20220217_1645091685.png" alt="image-20220217095443670" style="zoom: 67%;" />

Character motion is hard because we need to consider all three aspects:

- hyperparameter control
- balancing
- natural movement style

### dynamic 3d character motion techniques in VR

> http://www.lifesciencesite.com/lsj/life1003/127_19441life1003_846_853.pdf


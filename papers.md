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
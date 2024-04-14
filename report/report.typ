#import "@preview/lucky-icml:0.2.1": *

#show: icml2024.with(
  title: [Technical Report on Ray Tracing Algorithm Implementation],
  authors: (((name: "Jiaxin Huang",
   affl: ("hku"),
   email: "jiaxin.huang@connect.hku.hk"),), (
  hku: (
    department: [Department of Computer Science],
    institution: [Univeristy of Hong Kong],
    location: [Pokfulam],
    country: [Hong Kong SAR],
    ),)),
  keywords: ("Computer Graphics", "Ray-tracing"),
  abstract: [
    This report presents the implementation and experimental evaluation of a ray tracing algorithm employing the Phong illumination model for shading and lighting calculations. The algorithm supports diffusion, specular reflection, and refraction effects. It is accelerated using PyTorch and CUDA and achieves real-time rendering performance, generating antialiased high-resolution images in a few seconds. This report discusses the shading model and various optimization techniques used in this implementation. After that, the rendering performance is evaluated via a demonstrative scene.
  ],
  bibliography: bibliography("refs.bib"),
  header: [Technical Report on Ray Tracing Algorithm Implementation],
  accepted: none,
)

#set math.equation(number-align: bottom)

#let subfigure = figure.with(kind: "subfigure", supplement: [], numbering: "a", outlined: false)

#show ref: it => {
  if it.element != none and it.element.func() == figure and it.element.kind == "subfigure" {
    locate(loc => {
      let q = query(figure.where(outlined: true).before(it.target), loc).last()
      ref(q.label)
    })
  }
  it
}

#vruler(offset: -1.7in)

= Introduction

Ray tracing is a powerful rendering technique used in computer graphics to generate realistic images by simulating the behavior of light rays as they interact with different materials and geometry primitives @raytracing@shading. It provides realistic reflections, shadows, and refractions, leading to visually appealing and accurate results. Ray tracing technique has been widely adopted in various applications such as animation, gaming, and movie production.

The objective of this project is to implement the ray tracing algorithm and demonstrate its capabilities in rendering realistic scenes with various materials and lighting conditions. 

In this report, @sec:impl presents the implementation details of the ray tracing algorithm, including the shading models, ray generation mechanism, and acceleration techniques. @sec:demo covers the configuration and analysis of a demonstrative scene, exploring different material parameters and lighting conditions. Finally, @sec:conclu provides a summary of the achievements, limitations, and suggestions for future work.


= Implementation Details
<sec:impl>

Ray tracing is based on the principle of calculating intensities of light rays emitted by scene objects and reaching the camera's viewpoint, which may in turn depends on rays emitted directly by light sources or indirectly by other objects and reaching that object, leading to a recursive algorithm @raytracing.

== Ray Generation
The ray-tracing algorithm assumes a $(w times d)$-dimentional hypothetical canvas in front of the camera. To shade this canvas, we invert the directions of all light rays - letting $t$ rays emitting from the camera through each pixel on the canvas. In the case when $t>1$, the colors of such $t$ rays are averaged to produce an _anti-aliased_ pixel shading @distributed.

== Ray-Scene Intersection
For each out-going ray, the algorithm determines the closest object it hits by calculating the intersection distance with every object and taking the minimal.

== Shading and Lighting
Then, shading and lighting calculations are performed at the closest intersection point using the Phong illumination model @phong@phong2, which considers four lighting components: _ambient_, _diffuse_ and _specular reflections_ and _refractions (transmission)_. The Phong illumination model can be modified to account for both direct light sources and indirect lights emitted by other objects. The formula is as follows:

$ I &= I_i ( A + D + S + R)\
  A &= I_a\
  D &= k_d (sum_(op("light") l) (N dot L)I_l + integral_Omega (N dot omega) I_omega dif omega)\
  S &= k_s (sum_(op("light") l) (V dot L)^gamma I_l + integral_Omega (V dot omega)^gamma I_omega dif omega)\
  R &= k_r (sum_(op("light") l) (R dot L)^gamma I_l + integral_Omega (R dot omega)^gamma I_omega dif omega)\
$ <eq:phong>

where:
- $I$ is the final color of the intersection point;
- $I_i$ is the intrinsic color of the object at the intersection point;
- $I_a$ is the intensity of a hypothetical all-direction ambient light source;
- $I_l$ is the intensity of the incident light source $l$ (set to $0$ if the light is occluded by other objects, as to be discussed in @subsec:occlu);
- $I_omega$ is the intensity of rays emitted by other objects from direction $omega$ and reaching the intersection point (to be elaborated in @subsec:spawn). It's value is evaluated by applying Equation @eq:phong recursively;
- $k_d, k_s$ and $k_r$ are diffuse, specular reflection and refraction coefficients, respectively;
- $N$ is the surface normal at the intersection point;
- $L$ is the direction vector from the intersection point to the light source;
- $V$ and $R$ are the reflection and refraction direction vector of the incident eye ray, as to be determined in @subsec:ref_dir;
- $gamma$ is the glossiness coefficient.

=== Occlusion Detection
<subsec:occlu>
To create physically correct shadows, a ray is casted from the intersection point to the light souce when determining the $I_l$ term. If such ray intersects with any scene object, this light source is considered occluded and does not illuminate the intersection point. Therefore, $I_l$ is set to $0$.

=== Reflection and Refraction Direction
<subsec:ref_dir>

Given the incident light $I$ and surface normal $N$, the reflection $V$ and refraction light $R$ are
$ V = I + 2 c N $ and
$ R = eta I + (eta c - sqrt(1+eta^2(c^2 - 1)))N, $ where
$c=angle.l I, N angle.r$ is the cosine of the incident angle and $eta= eta_r/eta_i$ is the ratio of refraction indices of the incident and refraction media.

If $1+eta^2(c^2 - 1)<0$, there is not refraction light. This scenario is known as _total internal refraction_.

=== Spawning Recursive Rays
<subsec:spawn>
In Equation @eq:phong, one needs to evaluate the integral
$ I_op("spawned")=integral_Omega f(omega)I_omega dif omega, $
where $f(omega)$ is some coefficient. As it is computationally intractable to evaluate such integration directly, a Monte-Carlo method with _importance sampling_ is used to produce an efficient estimation.

Let $p(omega)$ be a probability distribution that is propotional to $f(omega)$:
$ p(omega)=f(omega)/(integral_Omega f(omega) dif omega). $
Then
$ I_op("spawned") 
&= integral_Omega f(omega)I_omega dif omega\
&= integral_Omega f(omega)/p(omega) I_omega p(omega)dif omega\
&prop integral_Omega I_omega p(omega)dif omega\
&tilde  1/M sum_(m=1)^M I_(omega_m),
$
where $omega_m$ are random variables sampled from $p(omega)$. Intuitively, this means we are more likely to sample rays who have larger contribution to the integration ($f(omega)$ being large).

Furthermore, determining the sample size $M$ is a matter of balancing between estimation accuracy and computational cost, espacially in a recursive algorithm where the sampling cost grows exponentially. Therefore, an adaptive strategy is used to determine the appropriate value of $M$. In general, $M$ is in the range $[1, 30]$ and is smaller when:
- the coefficient $k_{d,s,r}$ is small, because its contribution to the final rendered image is small;
- the glossiness $gamma$ is high, because spawned rays are highly concentrated and are likely to have small variance. In particular, when $gamma > 100$, the object is considered perfect reflector/transmitter and $M$ is fixed to 1;
- the recursion level is deep, which also means its contribution to the final rendered image is small.


== Programming Environment and Optimization

The ray tracing algorithm is implemented in Python, utilizing the PyTorch library for parallelization and CUDA for GPU acceleration. By leveraging the computational power of GPUs, the rendering performance is significantly improved. A typical rendering of a $1920 times 1080$ image with 2-level resursion and supersampling coefficient $t=9$, totaling 3.3 billion rays, takes only 6 seconds on a GeForce GTX 1080 Ti graphic card.

= Experimentation and Results
<sec:demo>

== Scene and Material Configurations
The examplar scene consists of three balls placed on a plane and surrounded by two walls. 
To evaluate the performance and visual quality of the algorithm, different material parameters are applied to the objects. The following variations are explored:

With labels in @fig:plain, @obj:matte simulates a matte plastic ball with $k_d=0.8, k_s=0.4$ and $gamma=4$, while @obj:metal is a glossy metal ball with $k_d=0.6, k_s=0.9$ and $gamma=100$. @obj:wall is a brown matte wall ($k_d=0.6, k_s=0.2$ and $gamma=50$) and @obj:gnd is half-matte-half-glossy ($k_d=0.3, k_s=0.8$ and $gamma=50$), allowing colors from neighboring objects subtly "bleed" onto its surface.

Finally, @obj:mirror is a perfact mirror and @obj:crystal is a transparent crystal ball with refraction index 1.25.


== Discussion of Results
The rendered images are shown in @fig. 

In @fig:plain, the scene is rendered with only plain shading. No illumination technique is used at all. 

In @fig:diffuse, a point light source is added above the scene and diffuse reflection is enabled. Notice the correct shadows below the three balls and behind the two walls. 

In @fig:specular, we test specular reflection only. Notice that different balls have different reflection radius because they have different glossiness parameter $gamma$.

In @fig:full, we put ambient, diffuse and specular reflection together, resulting in a reasonably realistic image. However, until now, object cannot reflect neighboring objects, because ray tracing is not yet enabled.

In @fig:trace, rays are spawned at the ray-scene intersection points to trace indirect lights emitted by other scene objects. Notice now that objects can leave a blurry reflection image on the ground because the ground has a low glossiness factor. There are also clear reflection images of neighboring objects on the surface of @obj:metal and @obj:crystal, as because their glossiness factor is high. Now the mirror can also correctly show the reflection of the scene. Finally, the artifacts on the pink crystal ball are due to refraction.


#let object(label) = {
  figure(kind: "object", supplement: [Object], numbering: "1", outlined: false)[
    #text(silver)[*#label*]
  ]
}

#let width = 15em

#figure(caption: [The rendered scene])[
  #subfigure(caption: [ Plain shading])[
    #image("/figs/plain.png", width: width)
    #place(dx: 7.9em, dy: -4.5em)[#object[1]<obj:matte>]
    #place(dx: 10.1em, dy: -3.6em)[#object[2]<obj:metal>]
    #place(dx: 12.6em, dy: -5em)[#object[3]<obj:wall>]
    #place(dx: 6em, dy: -2em)[#object[4]<obj:gnd>]
    #place(dx: 7em, dy: -6.9em)[#object[5]<obj:mirror>]
    #place(dx: 4.4em, dy: -4.8em)[#object[6]<obj:crystal>]
    ] <fig:plain>
  #subfigure(caption: [ Diffuse only])[
    #image("/figs/diffuse.png", width: width)] <fig:diffuse>
  #subfigure(caption: [ Specular only])[
    #image("/figs/specular.png", width: width)] <fig:specular>
  #subfigure(caption: [ Ambient, diffuse and specular])[
    #image("/figs/full.png", width: width)] <fig:full>
  #subfigure(caption: [ Ray tracing on])[
    #image("/figs/recursive.png", width: width)] <fig:trace>
]<fig>

= Conclusion
<sec:conclu>

The implementation of the ray tracing algorithm utilizing PyTorch and CUDA has demonstrated real-time rendering performance, successfully rendering a high quality image in a few seconds. The algorithm accurately simulates diffusion, specular reflection, and refraction effects, producing realistic images.

Despite the achievements, the implemented ray tracing algorithm has certain limitations, such as the lack of complex geometries and dealing with total refraction. Future work can focus on addressing these limitations and further optimizing the algorithm for improved efficiency and realism. Additionally, the algorithm can be extended to support mesh constructure and texture mapping.

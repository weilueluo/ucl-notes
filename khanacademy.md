# Khan Academy

- A line starts from camera at point C projected onto a plane at point P to an object O, can be represented as $f(t)=(1-t)C+tP$, as $t$ increases from $0$ to $1$, the line increases from $C$ to $P$.

  - Compute $t$ given $C$, $P$ and $O$.

    -  $t=\frac{|CO|}{|CP|}$.

  - Compute the intersection of line $CP$ with $O$, given the line equation of $O$ as $y=mx+c$.

    - $$
      \begin{align*}
      x&=t(|CP|_x)=L_x\\
      y&=t(|CP|_y)=L_y\\
      L_yt&=(L_ym)t+c
      \end{align*}
      $$

      In practice, $L_y$, $L_x$, $m$ and $c$ will be given, so we can solve for $t$.

    - There is a problem with this approach, that is when the line is vertical, then $m=\frac{\Delta y}{\Delta x}$ will divide by zero because there is no change on the $x$ axis, therefore we can multiple the line of equation by $\Delta x$ to avoid division by zero:
      $$
      \begin{align*}
      y&=\frac{\Delta y}{\Delta x}x+c\\
      \Delta xy&=\Delta yx+\Delta xc\\
      \Delta xy-\Delta yx-\Delta xc&=0\\
      \end{align*}
      $$
      


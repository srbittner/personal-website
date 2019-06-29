---
title: "Neuroeconomics: where microeconomics meets the information bottleneck"
published: true
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

A lasting challenge in economics research has been the incorporation of the (apparent) irrationality of human behavior into informative models of the exchange of goods and services. The fields of economics and psychology have always had a natural overlap, since this suboptimal behavior ultimately arises from the cognitive processes of humans interacting in the marketplace.  Recently, economists have started to dig deeper by looking to neuroscience for a mechanistic understanding of this suboptimal behavior.  In turn, neuroscientists are realizing that many of the classic formalisms and tools from microeconomics research – the focus of economics on the decisions of individuals --  may be useful for understanding neural computation and its associated mental effort.  This interdisciplinary field of research dedicated to human decision making is called *neuroeconomics*.

The growing desire for cross-adoption of methodology between these academic communities motivated the “[Workshop on Efficient Coding and Resource-Limited Cognition](https://econ.columbia.edu/event/workshop-on-efficient-coding-and-resource-limited-cognition/)” this past Friday, organized by Michael Woodford and the Columbia Economics Department held in the Jerome L. Greene Science Center, which houses Columbia’s Neuroscience Department.  The idea of this workshop title is that in the brain, there is a need for efficient coding of information in neural representations in such a biologically resource-limited setting (finite energy, finite neurons, temporal constraints, etc.).  How do these constraints shape neural processing and the resulting behavior?  A host of speakers from various fields and stages of their career gave talks on their research related to this topic.  Overall, the talks were quite impressive.  I appreciated the accessibility of the presentations for the general computational neuroscience audience, which might not remember what e.g. a marginal cost curve is. In this post, I’ll summarize the main ideas from the talks and discussion panels.

## Modeling mental effort in economics ##
Basic economic theories rely on the assumption that consumers and suppliers are behaving optimally given the prices of goods and services in the marketplace and their level of demand. A considerable amount of error in the predictions of these basic models is due to the suboptimality of human behavior, which is sometimes framed as “irrational” behavior.  However, if we consider the scientific evidence that making optimal decisions requires attention and the exertion of mental effort, and that this mental effort has an intrinsic cost, suboptimal decisions can actually be rationalized in the big picture (review, [Wooter & Botvinick, 2018](#Kool2018mental)).  Economic models that incorproate such a cost on mental attention are often called rational innatention (RI) models ([Sims, 2003](#Sims2003implications)).

In their review, Wooter and Botvinick outline three primary modeling strategies economists have taken to account for the cost of mental labor (Box 1).  In the first, most simply strategy, mental effort is cast as a negative additive component in a holistic utility function.  A concave cost vs mental effort curve would be akin to the idea of a cognitive law of diminishing returns.  In other words, the devotion of more cognitive resources to a particular decision becomes decreasingly rewarding as more cognitive effort has already been spent.  The second class of strategies models nonlinear effects of mental effort on total utility.  The third class of strategies considers that reward level may modulate the nonlinear effect of mental effort on utility.  

We can make a connection between this third class of models and classical labor supply theory (LST) in the figure below.

![a](/images/05_11_19/BotvinickAnalogy.png){:width="30%"}

([Wooter & Botvinick, 2018](#Kool2018mental))

In this plot, we are looking at how utility is modulated by leisure (inverse of effort) versus income. For a fixed budget of T hours (black diagonal line) and a concave utility surface, the employee maximizes utility by spending time on a mixture of work and leisure at the red point.  The concave utility surface has several other implications that predict real world behavior, such as employees working more/less when they have an income-compensated wage increase/decrease.  Analagously, there is experimental evidence that suggests utility is a concave function of reward and mental leisure ([Wooter & Botvinick, 2014](#Kool2014labor)).  The degree to which we spend time putting our brains to work versus relaxing them follows the same patterns as our chosen tradeoff between physical labor and leisure.

It certainly makes sense that by conserving mental energy, people are prone to suboptimal, seemingly irrational behavior.  However, this has highly nontrivial implications for neuroscience research.  This means that **decades of economic research and methodological development on labor supply theory should be directly applicable to studying the role of mental effort in neural computation**.  

As you may have guessed, LST is far from the only subfield of economics that can be directly repurposed for cognitive neuroscience.  Andrew Caplin, one of the keynote speakers, presented a novel method for measuring attentional cost from decision data using an RI model ([Caplin et al. 2018](#Caplin2018rational)).  His research group has identified a direct analogy between cost of production of a competitive firm, and the mental cost of decision making incurred by a consumer.  In microeconomics 101, we learn that the production cost of a firm is the area under the supply curve (left), and the remaining amount of total revenue is considered surplus, or the total advantage of having produced at level $$\bar{Q}$$.  Analgously, Caplin et al. constructed an incentive-based psychometric curve (IPC, right), which plots attentional incentives versus attentional output instead of output versus price.

![a](/images/05_11_19/CaplinIPC.png){:width="80%"} 

([Caplin et al., 2018](#Caplin2018rational))

**Mental cost can be recovered via integration of the IPC, just as production cost can be recovered for a competitive firm from its supply curve**.  This means that neuroscientists can design experiments in a way so that an IPC can be measured, and thus so can attentional cost.  Andrew's group is ramping up several research projects applying their IPC methodology to cases such as Mexican labor arbitration, decision support systems, warnings in autonomous driving systems, and false alert fatigue in medical diagnostics.

We also heard great economic research talks by Filip Matejka and Rafael Polania. Filip presented an RI theory of mental accounting -- why people make bad personal finance decisions related to purchasing subsitutes versus complements and naive diversification ([Koszegi et al. 2018](#Koszegi2018attention)).  Rafael presented some compelling theoretical explanations for why humans are bad at making consistent subjective value assessments [(Polania et al. 2019)](#Polania2019efficient).  All of this economic research relied heavily on information theory, most prominently the concept of the information bottleneck.

## The information bottleneck ##

In neuroeconomics, the idea of the information bottleneck is that (some info needs to get through a person and out the other side).  It was great to have NT open the workshop and explain this idea to us.  

![a](/images/05_11_19/informationBottleneck.png){:width="80%"} 

Seminal work on this idea in 2000 formalized this idea mathematically.  If you want to transmit something with a minimal distortion level, you would use this type of cost function. Instead, you may want your reconstruction term to also be information based (include theoretical justification)

P Talk bout colors 

P Talk about the role of mutual information term in all of the economics research.

Andrew Caplin stated that he thinks the most interesting and important future direction of neuroeconomics is to discover the compositional factors of measured mental cost.  


## Efficient representations in neural computation ##
So, where does this cost   

 Talk about hierarchical RL and SS's stuff

Hierarchical reasoning is slightly intangible for studying at level of circuit.  Look at what Alan did in visual cortex.


### References ###

<a name="Caplin2018rational"></a> Caplin, Andrew, Mark Dean, and John Leahy. *[Rational inattention, optimal consideration sets, and stochastic choice](https://academic.oup.com/restud/article/86/3/1061/5060717){:target="_blank"}*. The Review of Economic Studies 86.3 (2018): 1061-1094.

<a name="Kool2018mental"></a> Kool, W. & Botvinick, M. *[A Labor/leisure tradeoff in cognitive control](https://psycnet.apa.org/record/2014-30721-002)*. J. Exp. Psychol. Gen. 143, 131–141 (2014).

<a name="Kool2018mental"></a> Kool, Wouter, and Matthew Botvinick. *[Mental labour](https://www.nature.com/articles/s41562-018-0401-9){:target="_blank"}*. Nature human behaviour (2018): 1.

<a name="Koszegi2018attention"></a> Koszegi, Botond, and Filip Matejka. *[An attention-based theory of mental accounting](http://www.personal.ceu.hu/staff/Botond_Koszegi/mental_accounting.pdf){:target="_blank"}*. (2018).

<a name="Polania2019efficient"></a> Polania, Rafael, Michael Woodford, and Christian C. Ruff. [Efficient coding of subjective value](https://www.nature.com/articles/s41593-018-0292-0){:target="_blank"}*. Nature neuroscience 22.1 (2019): 134.


<a name="Sanborn2018representational"></a> Sanborn, Sophia, et al. *[Representational efficiency outweighs action efficiency in human program induction](https://arxiv.org/abs/1807.07134){:target="_blank"}*. arXiv preprint arXiv:1807.07134 (2018).


<a name="Sims2003implications"></a> Sims, Christopher A. *[Implications of rational inattention](https://www.sciencedirect.com/science/article/pii/S0304393203000291){:target="_blank"}*. Journal of monetary Economics 50.3 (2003): 665-690.

<a name="Tishby2015deep"></a> Tishby, Naftali, and Noga Zaslavsky. *[Deep learning and the information bottleneck principle](https://ieeexplore.ieee.org/abstract/document/7133169){:target="_blank"}*. 2015 IEEE Information Theory Workshop (ITW). IEEE, 2015.

<a name="Tishby2000information"></a> Tishby, Naftali, Fernando C. Pereira, and William Bialek. *[The information bottleneck method](https://arxiv.org/abs/physics/0004057){:target="_blank"}*. arXiv preprint physics/0004057 (2000).

<a name="Zaslavsky2018efficient"></a> Zaslavsky, Noga, et al. *[Efficient compression in color naming and its evolution](https://www.pnas.org/content/115/31/7937){:target="_blank"}*. Proceedings of the National Academy of Sciences 115.31 (2018): 7937-7942.


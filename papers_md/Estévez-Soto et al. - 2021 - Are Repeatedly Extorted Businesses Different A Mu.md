Journal of Quantitative Criminology (2021) 37:1115–1157 https://doi.org/10.1007/s10940-020-09480-8 

**<mark>ORIGINAL PAPER</mark>** 

# **Are Repeatedly Extorted Businesses Different? A Multilevel Hurdle Model of Extortion Victimization** 



### **Patricio R. Estévez‑Soto**<sup>**1**</sup> **· Shane D. Johnson**<sup>**1**</sup> **· Nick Tilley**<sup>**1**</sup> 

Published online: 9 October 2020 © The Author(s) 2020 

### **Abstract** 

**Objectives** Research consistently shows that crime concentrates on a few repeatedly victimized places and targets. In this paper we examine whether the same is true for extortion against businesses. We then test whether the factors that explain the likelihood of becoming a victim of extortion also explain the number of incidents suffered by victimized businesses. The alternative is that extortion concentration is a function of event dependence. **Methods** Drawing on Mexico’s commercial victimization survey, we determine whether repeat victimization occurs by chance by comparing the observed distribution to that expected under a Poisson process. Next, we utilize a multilevel negative binomial-logit hurdle model to examine whether area- and business-level predictors of victimization are also associated with the number of repeat extortions suffered by businesses. 

**Results** Findings suggest that extortion is highly concentrated, and that the predictors of repeated extortion differ from those that predict the likelihood of becoming a victim of extortion. While area-level variables showed a modest association with the likelihood of extortion victimization, they were not significant predictors of repeat incidents. Similarly, most business-level variables significantly associated with victimization risk showed insignificant (and sometimes contrary) associations with victimization concentration. Overall, unexplained differences in extortion concentration at the business-level were unaffected by predictors of extortion prevalence. 

**Conclusions** The inconsistent associations of predictors across the hurdle components suggest that extortion prevalence and concentration are fueled by two distinct processes, an interpretation congruent with theoretical expectations regarding extortion that considers that repeats are likely fueled by a process of event dependence. 

**Keywords** Repeat victimization · Hurdle model · Extortion · Organized crime · Crimes against businesses 

> * Patricio R. Estévez-Soto patricio.estevez@ucl.ac.uk 

> 1 Department of Security and Crime Science, University College London, 35 Tavistock Square, London WC1H 9EZ, UK 

Vol.:(0123456789)1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1116 

## **Introduction** 

Decades of research suggest that crime is concentrated on a small proportion of places (Lee et al. 2017) and victims (SooHyun et al. 2017). In fact, the patterns observed have been so consistent that Weisburd (2015) has recently coined the term “the law of crime concentration at place.” However, it is unclear how universal this empirical pattern might be across countries, as well as crime and target types, as most research has focused on the US and Canada, a handful of European cities, and Australia<sup>1</sup> ( e.g. Andresen et al. 2016, 2017; Curman et al. 2015; Farrell et al. 2005; Sagovsky and Johnson 2007; Tseloni et al. 2004; Perreault et al. 2010; Lynch et al. 1998; Johnson and Bowers 2010), and on “traditional” crimes against individuals and households<sup>2</sup> (e.g. Daigle et al. 2008; Osborn and Tseloni 1998; Tseloni et al. 2002; Tseloni and Pease 2004, 2003; Johnson et al. 1997; Tseloni et al. 2004; Kleemans 2001; Young and Furman 2007). 

In particular, there is a notable scarcity of research that has systematically studied the concentration of organized crimes,<sup>3</sup> an area of study that could benefit considerably from more rigorous quantitative assessments (Sansó-Rubert Pascual 2017, p. 29–30). Thus, this study is concerned with understanding the concentration patterns of extortion—an archetypal organized crime (Tilley and Hopkins 2008, p. 449). Specifically, it examines patterns of extortion among Mexican businesses to determine if concentration occurs, and if it does, to determine the factors that may explain it. 

Current research suggests that crime concentration on specific targets (henceforth “repeat victimization”) is driven by two mechanisms: _Risk heterogeneity_ (Pease 1998; Johnson 2008 ) proposes that enduring differences in target characteristics make some targets more attractive than others; while _event dependence_ (Pease 1998; Johnson 2008) suggests that the risk of victimization is dynamic, with the risk to victimized targets increasing—at least temporarily—following an initial offense. Though studies have found that both mechanisms have a part to play (e.g. Johnson 2008; Lauritsen and Davis Quinet 1995; Pitcher and Johnson 2011; Lynch et al. 1998; Tseloni and Pease 2004, 2003), it is generally assumed that the risk factors associated with victimization _prevalence_ —the likelihood of becoming a victim—also explain its _concentration_ —the number of incidents per victimized target (Pease and Tseloni 2014, p. 31). Thus, analytic studies generally focus on explaining _incidence_ —the number of incidents per potential target—using a single set of predictors to examine the entire distribution of crime, rather than examining whether the predictors that differentiate victims from non-victims also explain the amount of crime suffered by victimized targets (Pease and Tseloni 2014, p. 31). 

However, this assumption is largely based on previous findings concerning household property crimes (Osborn et al. 1996), and it is unlikely to apply in the case of extortion. To explain, like crimes such as personal fraud (see Titus and Gover 2001, p. 135), extortion requires the victim’s cooperation for the offender to succeed (Best 1982, p. 109), thus repetition may be influenced by a victim’s level of cooperation (which can only be observed 

> 1 Only a small number of recent studies have examined patterns in radically different contexts such as Brazil (Melo et al. 2015), Malawi (Sidebottom 2012), Taiwan (Kuo et al. 2012) and South Korea (Park 2015). 

> 2 For exceptions see Andresen et al. (2017), Bowers et al. (1998), van Dijk and Terlouw (1996), Salmi et al. (2013), Burrows and Hopkins (2005), Gill (1998), Hopkins and Tilley (2001), Matthews et al. (2001), and Yu and Maxfield (2014). 

> 3 Welcome exceptions are the studies on kidnapping for ransom in Colombia by Pires et al. (2014) and Stubbert et al. (2015), and Dugato’s (2014) study on bank robberies in Italy. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1117 

through interaction), rather than by a stable set of characteristics. Furthermore, extortion is often characterized as a long-term relationship between victims and offenders (Elsenbroich and Badham 2016; Kelly et al. 2000, p. 64); thus, repetition may be unaffected by stable risk factors once an extortive relationship has been established. 

To examine this, this study uses data from Mexico’s national commercial victimization survey to trial a novel modeling strategy—the multilevel negative binomial-logit hurdle model—to assess whether the predictors associated with the likelihood of extortion victimization are also associated with the number of repeat extortions suffered by businesses. If predictors are inconsistent across the two measures, this would suggest that extortion concentration is fueled by a process distinct from that which might explain extortion prevalence. Thus, this study contributes towards expanding our understanding of micro-level patterns of crime concentration in two ways. First, to our knowledge, this study represents the first application of a multilevel negative binomial-logit hurdle model to study crime concentration,<sup>4</sup> highlighting its potential usefulness to study other crime types where repetitions are thought to be driven by distinct processes—such as domestic violence<sup>5</sup> (Biderman 1980; Rand and Saltzman 2003). Second, the study contributes to the literature on crime concentration by examining whether or not the patterns consistently observed elsewhere also apply to: a) a crime type that has received little research attention, and b) a country that has so far been neglected in the literature. 

The article proceeds as follows. In the next section we briefly introduce the background of extortion in Mexico. Next, we review the literature on modeling repeat victimization, and on the predictors of extortion victimization. The next section covers the data and analytical strategy employed, followed by the research findings and a discussion of their implications and limitations. 

## **Background: Extortion in Mexico** 

Extortion is the third most common crime against Mexican businesses, after petty theft and robbery (INEGI 2014a). However, extortion is very prominent in the public agenda, as it has been linked to notorious episodes of violence. During the first two months of 2011 there were 119 arson attacks linked to extortion in the infamous Ciudad Juárez<sup>6</sup> (GuerreroGutiérrez 2011). Later that year, an arson attack against a casino that refused to comply with extortion demands in Monterrey (a city in northeast Mexico) killed 52 people, making it one of the deadliest criminal incidents in Mexico’s recent history (Corcoran 2012; Wilkinson 2011). Lastly, widespread extortion of farmers in 2012 and 2013 led to a politically destabilizing uprising of _autodefensas_ —self-defense paramilitary groups—in many rural areas of the country (Shirk et al. 2014, p. 10). 

> 4 However, the approach has been used in crime research to study sentencing (Hester and Hartman 2017; Rydberg et al. 2017), intimate partner violence (Hellemans et al. 2015), specific types of homicides (Guerrero-Gutiérrez 2011; Baller et al. 2009), and the influence of incarceration on health care availability (Wallace et al. 2015). 

> 5 Though Hellemans et al. (2015) used hurdle models in a study of intimate partner violence (IPV), their study is focused on the relationship between lifetime experience with IPV and victims’ relational and sexual well-being, and does not address the factors associated with IPV repeat victimization. 

> 6 A city on the Mexico-U.S. border, across from El Paso, Texas. In 2010, Ciudad Juárez was the most violent city in the world (Redacción 2010), with a murder rate of 216 per 100,000 inhabitants (Rios 2012). 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1118 

Over the past two decades, Mexican organized crime has been transformed from a few powerful groups focused mainly on drug trafficking, to a plethora of small—though far more violent—organizations involved in drugs, extortion, kidnapping, murder and other crimes (for reviews, see Bunker 2013; Shirk and Wallman 2015). Three broad hypotheses have been proposed to explain this transformation. The first focuses on how Mexico’s transition to democracy<sup>7</sup> —after 71 years of one-party-rule—dissolved informal political control mechanisms that kept organized crime in check, leading to violent power struggles for control of lucrative criminal markets (Rios 2015; Duran-Martinez 2015; Dell 2014; Astorga 2012; Aguirre and Herrera 2013). The second links changes in international drug 2013; Rios 2012; Paul et al. 2011; markets to conflicts over strategic territories (Corcoran Brophy 2008). Finally, the third focuses on the instability created by the Mexican government’s crackdown on organized crime from 2006 onwards (Rios 2012; Calderon et al. 2015; Correa-Cabrera et al. 2015; Jones 2013; Dickenson 2014; Osorio 2015). 

These approaches, concerned as they are with broad manifestations of organized crime, may explain trends in organized crime violence in Mexico—and by extension in extortion—at the national, sub-national, or even city level. However, they are inadequate to provide insight into patterns at the micro-level of places (in this case, businesses), which is the focus of this study. 

## **Repeat Victimization** 

Repeat victimization (RV) occurs when a target (however defined) suffers the same offense two or more times during a given time period (Grove and Farrell 2010; Pease 1998). In a seminal study, Farrell and Pease (1993) noted that according to the 1982, 1988, and 1992 sweeps of the British Crime Survey, repeat victims suffered between 71% and 81% of all incidents, yet they amounted to only 14–20% of respondents. International evidence is consistent with these findings. Based on results from 17 countries originally reported by Farrell and Bouloukos (2001), Farrell and Pease (2011) estimate that around “40 per cent of crimes against individual people and against households are repeats ... with variation by crime type and place” (p. 123). 

Patterns of RV were first identified during the 1970s with the advent of victimization surveys. These were instituted to measure the extent of crime while overcoming the fact that many crimes were not reported to (or recorded by) the police (Gottfredson 1986; Sparks 1981b; Wetzels et al. 1994). Surveys provide two measures of crime: the number of victims (prevalence), and the number of crime incidents (incidence)—both usually expressed as per capita rates. The ratio of incidence to prevalence (concentration) provides a crude summary of repeat victimization. However, these summary measures are insufficient to grasp the extent of concentration, as they ignore the wide disparity in crime risks experienced across the population. Thus, the extent of crime concentration is better represented by the frequency distribution. Early studies that analyzed such distributions demonstrated that concentration was unlikely to be the product of chance (Hindelang et al. 1978, p. 125–149; Sparks et al. 1977, p. 88–106). This was demonstrated by comparing the observed frequencies to those generated by a Poisson<sup>8</sup> process, which indicated that 

> 7 For discussions on the transformation of Mexican democracy, see Camp (2012) and Smith (2012). 

> 8 Events generated by a Poisson process are _random_ and _independent_ insofar as they occur at a constant rate ( _휇_ ) not affected by past events (Sparks 1981a). 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1119 

there were far more repeat victims than those expected (Sparks et al. 1977; Hindelang et al. 1978). These non-random patterns, in turn, suggested that _either_ events were not independent—and concentration was due to a contagion-like process—or that the population could not be characterized by a constant rate—meaning that the some individuals faced different risks of victimization (Sparks 1981a, p. 766–767). More recent research which has explicitly examined risk heterogeneity and event dependency, however, suggests that both mechanisms contribute to repeat victimization, and that the processes act in concert (see Johnson 2008; Pease 1998; Pitcher and Johnson 2011). 

Empirical backing for the first mechanism—known as “boosts” (Pease 1998), “state dependence” (Lauritsen and Davis Quinet 1995; Osborn and Tseloni 1998), and “event dependence” (Johnson 2008, the term we use hereafter)—comes from longitudinal studies that have found that victimizations suffered in previous periods increase the risk of suffering crimes in the future, even after controlling for stable risk factors (e.g. Lauritsen and Davis Quinet 1995; Lynch et al. 1998; Tseloni and Pease 2004, 2003). Further evidence for event dependence is found in the temporal and spatial patterns of repeat and near-repeat victimization (see Morgan 2001), which show temporary increases in risk to victimized 1991; targets and those in their vicinity shortly after an offense has taken place (Polvi et al. Johnson et al. 1997; Johnson and Bowers 2004; Johnson et al. 2009). On the other hand, evidence for the second mechanism—known as “flags” (Pease 1998), “population heterogeneity” (Osborn and Tseloni 1998; Nelson 1980; Lauritsen and Davis Quinet 1995; Wittebrood and Nieuwbeerta 2000), and “risk heterogeneity” (Johnson 2008, the term we use hereafter)—is found in studies that have analyzed how individual and contextual character1990; Miethe istics are associated with differing risks of victimization (Miethe and Meier and McDowall 1993; Lauritsen 2010; Lauritsen and Rezey 2018). 

In terms of theory, both mechanisms are underpinned by environmental criminology (Bouloukos and Farrell 1997; Farrell and Pease 2014; Wartell and Gallagher 2012). The rational choice perspective (Cornish and Clarke 1985; Clarke and Cornish 2017) explains why a) certain characteristics make some targets appear more attractive or vulnerable to a wide range of offenders, as they speak to the effort, risks, and rewards that offenders consider when engaging in a crime (Cornish and Clarke 1987; Clarke and Cornish 2017), and also b) how knowledge gleaned from past crimes influences future target selection (Cornish and Clarke 1985). On the other hand, the routine activity approach (Cohen and Felson 1979; Felson 2017) and crime pattern theory (Brantingham et al. 2005; Brantingham and Brantingham 1993) explain how human ecology and urban topology shape the convergence of suitable targets and likely offenders in the absence of capable guardians, and thus provide explanations for a) the heterogeneous distribution of risk (Grove and Farrell 2010; Maxfield 1987b), as well as b) the constraints imposed by the underlying ecological and urban structure to temporary increases in crime risk following an initial event (Johnson et al. 1997; Sagovsky and Johnson 2007; Rosser et al. 2016). 

### **Modeling Repeat Victimization** 

In addition to measuring crime, victimization surveys capture a wealth of information regarding respondents and their environments, such as demographic and socioeconomic characteristics, the presence of protective and security equipment, lifestyle details, perceptions and fear of crime, and measures of neighborhood disorder (UNODC/UNECE 2010; Cantor and Lynch 2000; Pease and Tseloni 2014). Such information has provided opportunities to investigate the potential associations between victimization risks and target (as 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1120 

well as contextual) characteristics (e.g. Hindelang et al. 1978; Sparks et al. 1977; Miethe et al. 1987; Tseloni et al. 2004). 

However, most early approaches to modeling victimization focused on the prevalence of victimization. These studies applied logistic regression to identify the factors associated with the risk of being victimized (e.g. Maxfield 1987a; Miethe et al. 1987). Such approaches, however, disregard the differential risks associated with repeat victimization (Pease and Tseloni 2014, p. 35). Pease and Tseloni (2014) ascribe this to the fact that the significance of repeat victimization had been somewhat neglected, and to a lack of accessible statistical tools and techniques suitable for the analysis of count data (p. 35). 

Crime incidents are recorded as discrete counts with a lower bound of zero, requiring special modeling frameworks such as the Poisson (e.g. Berk and MacDonald 2008; MacDonald and Lattimore 2010). Furthermore, incidence data tend to be heavily right-skewed, with many observations for zero incidents and a long tail with few observations for targets that suffered many incidents. This leads to overdispersion, which occurs when the variance of a distribution exceeds its mean (Cameron and Trivedi 2013, p. 4; Hope and Norris 2012, p. 544). The implication of this for modeling is that the standard Poisson model (which assumes that the mean of a distribution equals its variance) can generate erroneous standard errors (Rydberg and Carkin 2016, p. 63). Thus, the preferred distribution to model crime incidence is the negative binomial (see Pease and Tseloni 2014; Tseloni 1995). This allows overdispersion to be incorporated via a dispersion parameter (see the “Analytical Strategy” section), which captures unexplained differences in crime incidence between two targets that are otherwise identical in terms of the covariates included in the model (hereafter “unexplained heterogeneity”, Osborn and Tseloni 1998). This modeling approach has been further strengthened by the use of multilevel<sup>9</sup> models (e.g. Goldstein 2011) which can incorporate hierarchically structured and repeated measures data and can help (in the context of crime and place) distinguish how much unexplained heterogeneity can be attributed to area and individual-level sources (for a seminal study, see Tseloni 2006). 

With this modeling framework, studies test the risk heterogeneity hypothesis by incorporating covariates thought to affect a target’s expected victimization incidence. For example, in a study on burglary victimization across Europe, Tseloni and Farrell (2002) used a multilevel negative binomial model to investigate the effects of household and countrylevel characteristics on burglary incidence in eight European countries. Controlling for the effect of previous victimizations, the study found significant sources of risk heterogeneity consistent with the routine activities theory: absence of guardianship (e.g. being single or divorced), being close to likely offenders (as measured by household poverty), and increases in target attractiveness (e.g. car ownership as a sign of household affluence) were associated with increases in burglary incidence, to cite a few examples (Tseloni and Farrell 2002, p. 156–157). Furthermore, the study also found there was essentially no unexplained heterogeneity due to between-country differences, meaning that unexplained differences in burglary risk were likely due to target and subnational (e.g. city or region) level differences (Tseloni and Farrell 2002, p. 155–156). 

Event dependence, on the other hand, can be tested by incorporating the temporal dimension using longitudinal data (Lynch et al. 1998, p. 15). While Tseloni and Farrell (2002) included measures of previous victimization experiences, the cross-sectional nature of the data they used (the International Crime Victims Survey, see Mayhew and van Dijk 

> 9 Terminology varies according to specific disciplines, but multilevel models are also known as mixed effects models and hierarchical linear models (Bell et al. 2008, p. 1112). 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1121 

2014 ) impeded their ability to explicitly model the effects of event dependence on crime incidence (see Lynch et al. 1998). Studies that have employed longitudinal victimization data (e.g. repeated interviews with the same target), have found that victimization experiences in earlier observations are significant predictors of victimizations in posterior observations, even after controlling for risk heterogeneity (e.g. Lauritsen and Davis Quinet 1995; Lynch et al. 1998; Tseloni and Pease 2004, 2003). 

Most victimization surveys employ cross-sectional designs (Lynch 2006, p. 249; Mayhew and van Dijk 2014, p. 2604). In practice, this means that models based on cross-sectional data—the type of data used in this study—cannot distinguish between event dependence and risk heterogeneity, lumping the effects of the former with those of unexplained heterogeneity (Osborn et al. 1996; Osborn and Tseloni 1998; Pease and Tseloni 2014; Heckman 1981). 

A potential workaround initially considered to explore the effect of event dependence in the absence of longitudinal data was to assess whether the factors that explain one-time victimizations differ from those that explain repeated incidents. In a seminal paper, Osborn et al. (1996) employed a “double hurdle” bivariate probit model to compare the transition probabilities from non-victim to victim and from one-time victim to repeat victim for household property crimes, but found that the predictors of first and repeated victimizations were generally the same (Osborn et al. 1996, p. 243). This finding has been influential in subsequent studies and is widely cited as a justification to use a single set of predictors to explain the entire distribution of crime incidents (e.g. Tseloni et al. 2002, p. 113; Pease and Tseloni 2014, p. 31; Tseloni and Pease 2014, p. 5). 

This consensus fails to consider that the effect and relative contributions of risk heterogeneity and event dependence to repeat victimization may vary considerably across different crime types (Johnson 2008, p. 236). One of the most parsimonious explanations for event dependence draws from the fact that offenders often return to victimize past targets (Bernasco 2008; Everson and Pease 2001), suggesting that the choice of future tar2014; Bernasco 2008; Johnson et al. gets is influenced by previous experience (Johnson 2009). It follows that the contribution of event dependence will likely be higher for crimes where “the effort and/or risk of a second offence is clarified by victim response to a first offence” (Farrell et al. 1995, p. 396). For example, in a study of bank robbery, Matthews et al. (2001) found that success in past robberies was positively associated with future incidents—the amount stolen in past incidents adequately predicting future risk. Thus, as extortion is a crime where success depends on a victim’s willingness (reluctance) to cooperate (Best 1982, p. 109), the manner in which the victim responds may have a strong bearing on an offender’s decision to repeat the offense against the same target. For example, compliance in one incident could beget further victimizations as the victim is known to be lucrative and responsive. 

Furthermore, the importance of event dependence is likely to be decisive in crimes where repeated victimizations are the product of an ongoing relationship between victims and offenders, such as recurrent violent episodes framed within an abusive relationship (Biderman 1980, p. 29; Rand and Saltzman 2003). It is likely that extortion falls within this category of offenses, as repeated extortions are often characterized as an ongoing condition (Biderman 1980, p. 29; Elsenbroich and Badham 2016; Kelly et al. 2000, p. 64). Thus, concentration on repeatedly extorted targets may be unaffected by risk heterogeneity and instead indicate that such an institutionalized relationship exists. Therefore, it is conceivable that the risk factors affecting extortion prevalence may be distinct from those that affect extortion concentration, which warrants a modeling strategy that is able to differentiate such mechanisms. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1122 

### **A Hurdle Model of Extortion Victimization** 

The hurdle model (Mullahy 1986; Cameron and Trivedi 2013) is a suitable alternative for distinguishing the factors that influence prevalence from those that influence concentration. These models (unrelated to the “double hurdle” bivariate probit used by Osborn et al. 1996) combine two processes: one that generates positive counts ( ≥ 1 ) versus zero counts ( = 0 ); and another that generates only positive counts ( ≥ 1 ) (Hilbe 2011 , p. 355). The first process corresponds to the prevalence risk and is usually modeled using logistic regression, whereas the second estimates concentration using a truncated count model—usually the truncated-at-zero negative binomial, as the distribution remains overdispersed (Cameron and Trivedi 2013). 

1992 An alternative is the zero-inflated model (Lambert ). Zero-inflated models are similar to the hurdle framework insofar as they consider that counts are produced by a mixture of two distinct mechanisms. However, these types of model were developed to handle a specific problem encountered with some data—excess zeroes (e.g. Hilbe 2011, p. 355; Park and Fisher 2015, p. 1138; Tseloni and Pease 2014, p. 22). Rather than explicitly distinguishing between the processes that lead to prevalence and concentration, zero-inflated models consider that there is one process granting “immunity” to some targets, and a second that determines the incidence of victimizations that nonimmune targets experience (e.g. Park 2015; Park and Fisher 2015). Crucially, the process generating counts does not only estimate positive counts, but zero counts as well. The latter correspond to targets that, although not deemed to be statistically immune to victimization, did not experience any incidents during the period sampled. Thus, given that zero-inflated models do not explicitly distinguish prevalence from concentration, they cannot determine whether predictors are constant across both measures, and as such are unsuitable for our purposes here. Furthermore, the data used in the current study showed no signs of zero-inflation when compared to a negative binomial expectation (see the “Univariate Analysis” section), and hence would also be unsuitable for this reason. 

In contrast, hurdle models first assess the risk of victimization prevalence across all targets, and then estimate the concentration of incidents experienced by victimized targets. If victimization patterns are taken as an indirect measurement of offender-decision making (Hough 1987), then the hurdle model allows us to indirectly test whether the factors that influence victim-selection decisions are distinct from those that affect the decision to target past victims. Such an interpretation would be consistent with the effect of event dependence discussed above, where the decision to commit a repeat extortion is associated with the outcome of the first extortion attempt. Similarly, the hurdle model can also be seen as a more appropriate model of an extortive relationship, with predictors for prevalence explaining the risk of being initially targeted for an extortive relationship, and the predictors for concentration explaining how much exploitation businesses can expect once the relationship has been established. 

In any case, to empirically assess if extortion exhibits a pattern of repeat victimization consistent with the hurdle framework, the observed distribution of extortion must first be shown to exceed chance expectation, as some level of repeat victimization is to be expected by chance. Thus, our first hypothesis is: 

- **H1:** There are significantly more repeat extortion incidents than would be expected 

- on the basis of random victimization. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1123 

If repeat extortion is found to be non-random—and given that the role of event dependence is expected to be more prominent in the case of extortion, and that extortion often leads to enduring victim-offender relationships—we expect that once a business is victimized, extortion concentration will not be consistently associated with the predictors of extortion prevalence. 

- **H2:** Once a business is extorted, the predictors that explain extortion concentration are different from those that explain extortion prevalence. 

## **Predictors of Extortion Victimization** 

Broadly, extortion is demanding something—such as goods, services, property and especially money—through intimidation, e.g. by threatening violence or other harms (extortion 2010; Savona and Sarno 2014; Elsenbroich and Badham 2016; extortion 2017). The extortion of businesses is often considered a quintessential activity of organized crime groups (von Lampe 2005, p. 235; Paoli 2014, p. 7; Campbell 2013, p. 32; Konrad and Skaperdas 1998, p. 462; Frazzica et al. 2013, p. 99; Schelling 1971, p. 646–647; Tilley and Hopkins 2008, p. 449). Yet, the literature on extortion has rarely been concerned with understanding the phenomenon at the incident level, focusing instead on broader manifestations of institutionalized extortion,<sup>10</sup> which involves “the continuous, regular and systematic extortion of several victims” (Elsenbroich and Badham 2016). 

Consequently, there has been scant research on the micro-level factors associated with extortion victimization. Attention has focused instead on macro-level factors that are thought to be associated with widespread extortion ( e.g. Savona and Zanella 2010; TRANSCRIME 2009). Therefore, the literature so far makes few suggestions regarding micro-level variables that may be useful as predictors of extortion risk. In this section we provide a brief overview of the macro and micro level factors that have been thought to affect extortion victimization. It is important to note that this review does not aim to list all possible sources extortion risk, but to identify variables that may be used to test H2. 

### **Macro‑level Influences** 

Research on extortion by organized crime emphasizes the importance of macro, area-level influences. For example, Kleemans (2018) asserts that the level of aggregation relevant to study extortion is not “a specific point in space where an offender meets a target,” but broader “territories” controlled by organized crime groups (p. 874). Though no study has examined the effect of area-level characteristics on _repeat_ extortion, it is possible to identify three potential predictors: 

- _Rule of law:_ According to protection theory (see Kleemans 2015, 2018), extortion is an outcome of “alternative governance” structures imposed by criminal groups absent strong 

> 10 Such phenomena are variously labeled as racketeering (McIntosh 1973), extortion racketeering (Savona and Sarno 2014; Savona and Zanella 2010), extortion racket systems (Frazzica et al. 2013; La Spina et al. 2014), private protection (Gambetta 1993; Varese 2001), and violent entrepreneurship (Volkov 2002), among others. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1124 

legitimate governance (Kleemans 2018), whereby groups “tax” businesses operating in their territories in exchange for “protection” (Gambetta 1993; Paoli 2002; Kleemans 2014; Varese 2014). Thus, a potential predictor of extortion risk at the macro-level is the “strength” of legitimate governance structures across Mexican states (Sung 2004; Skaperdas 2001). 

- _Corruption prevalence:_ Researchers have also noted that the absence or presence of legitimate governance is not the only relevant variable, its quality must also be considered. Thus, studies have found that widespread predatory corruption (a marker of bad governance) may be an important predictor of increased extortion risks (e.g.  DíazCayeros et al. 2015; Tulyakov 2001; Morris 2013; Frye and Zhuravskaya 2000). 

- _Nature of local organized crime groups:_ The term “organized crime” is an umbrella term used to group a wide variety of criminal structures, activities and extra-legal governance arrangements (von Lampe 2016; Paoli and Vander Becken 2014). In particular, not all organized crime groups engage in the same types of criminal activities. Recent research classifies Mexican criminal groups into two types: a) those that are mostly focused on drug trafficking, and b) those that are more reliant on violence and Mafiastyle “protection” markets—i.e. extortion. (Jones 2016; Corcoran 2013; GuerreroGutiérrez 2011). Thus, it would be expected that: 

   - (a) areas where drug trafficking-focused organized crime groups operate may be associated with lower extortion risks, and 

   - (b) areas where violent, mafia-style organized crime groups operate may be associated with higher extortion risks. 

### **Micro‑level Influences** 

The literature on repeat victimization has consistently found that area crime rates mask substantial within-area heterogeneity. Indeed, in the case of extortion Savona and Sarno (2014) and La Spina et al. (2014) note that victim selection is not random, but is instead guided by victim vulnerability (see also Savona 2012, p. 8). While no research exists that has systematically analyzed business-level explanations for extortion risk in the Mexican context, findings from other contexts suggest predictors for use in this study: 

- _Corruption victimization:_ While corruption at the macro-level suggests an indirect relationship between the quality of governance and extortion risk, the often close relationship between organized criminals and government officials in Mexico (Díaz-Cayeros et al. 2015, p. 255–256; Morris 2013) suggests that a direct relationship between extortion and corruption victimization at the micro-level is also plausible. Alternatively, a relationship at the micro-level could be due to the presence of a variable independently affecting extortion and corruption risk—e.g. a vulnerability that separately attracts both extortionists and corrupt officials. This would be consistent with findings on multiple victimization that suggest not only that a) specific types of crime are concentrated on a small subset of targets, but also that b) those who repeatedly suffer one type of crime are more likely to suffer other types (Tseloni et al. 2002). 

- _Business age:_ The literature suggests that older businesses may be somewhat protected from extortion due to long-standing ties developed in their communities (and hence with organized criminals that operate there) (Varese 2011, 2014), and that new busi- 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1125 

nesses may inadvertently attract organized criminals by drawing attention to themselves through opening ceremonies or advertising (Chin et al. 1992; Chin 2000). 

- _Business type:_ Research on extortion by the Italian Mafia (e.g. Di Gennaro and La Spina 2016; La Spina et al. 2014; Frazzica et al. 2013) and Chinese gangs (e.g. Chin 2000; Kelly et al. 2000; Chin et al. 1992) suggests that some business types—particularly restaurants, hotels and bars—are at especially high risk of extortion. It is assumed that these business types are particularly at risk due to being inherently vulnerable to intimidation (Schelling 1971, p. 648–649). 

- _Business size:_ Several studies have found business size to influence crime risk in gen- 

- eral and extortion in particular, however the precise mechanism behind this relationship is unclear. Gill (1998) notes that small businesses in the UK suffer disproportionally more crime, which may suggest that risk is related to victim vulnerability. Yet, there is also evidence pointing to elevated extortion risks for larger businesses (Broadhurst et al. 2011; Kelly et al. 2000), which suggests that extortionists might select victims on the basis of potential rewards. We speculate that there is a trade-off between vulnerability and profitability: while smaller business may be more vulnerable, they offer fewer potential rewards making them less attractive to extortionists. 

In the next section, we discuss the data analyzed and the analytical strategy adopted. 

## **Methodology** 

### **Data** 

The primary data analyzed are from the 2014 sweep of Mexico’s nationally representative commercial victimization survey, the _Encuesta Nacional de Victimización de Empresas_ (ENVE, INEGI 2014c). This is, to our knowledge, the largest sample survey of business crime victimization that has been conducted hitherto and provides a rare opportunity to subject extortion patterns to systematic quantitative analysis. The survey is conducted biennially, sampling all business sectors except agriculture and the public sector. The first part of the survey, the main questionnaire, records the prevalence and incidence of crimes suffered by respondents during the previous calendar year (in this case, 2013). The victim form focuses on information concerning each incident of victimization  (though capped at 7 victim forms per crime type per victim, Jaimes Bello and Vielma Orozco 2013). We analyzed responses captured in the main questionnaire, as these offer a readily available uncapped summary of victimization experiences<sup>11</sup> (Trickett et al. 1992; Farrell and Pease 1993). 

The sample consisted of 33,479 business premises stratified and randomly selected from INEGI’s National Statistical Directory of Economic Units<sup>12</sup> ( _Directorio Estadístico Nacional de Unidades Económicas_ , DENUE, with 3.7 million units) (INEGI 2014b). Responses were collected through face-to-face interviews with the highest-ranking person 

> 11 To mitigate the risk of misclassification, interviewers provide respondents with an index card detailing the different crime types and their non-legal definitions (INEGI 2014c). 

> 12 The sampling unit for all business types except mining, transport and construction was premises; in the exceptions, the unit was the business (INEGI 2014b). Stratification was based on business size and the state in which the premise was established. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1126 

in micro and small businesses, and with security or finance managers in medium and large businesses. Computer assisted telephone interviewing (CATI) was used to follow up incomplete questionnaires (Jaimes Bello and Vielma Orozco 2013), leading to a response rate of 84%. 

Access to the anonymized individual responses collected by the ENVE is restricted by INEGI, as the combined characteristics of each respondent could potentially be used to infer their identity. This means that all analyses presented herein had to be performed remotely with no interactive access to the data. To overcome this constraint, all analyses were conducted using automated R scripts<sup>13</sup> —which provide the additional benefit of ensuring reproducibility—and remotely processed by INEGI staff in Mexico City on a Windows platform and R (R Core Development Team 2015). Downsides of this are that we had very limited control over the computational resources available (including software availability), and that analyses took longer to complete (as they could not be conducted interactively). 

### **Dependent Variable** 

The dependent variable is the number of incidents of extortion<sup>14</sup> suffered by surveyed businesses. The prevalence rate was 80.46 victims per 1000 businesses, whereas the incidence rate was 132.77 incidents per 1000 businesses. The concentration rate was thus 1.65 extortions per victim. However, the distribution of extortion, shown in Table 1, reveals that extortion victimization is far more concentrated than such summary statistics would suggest. Repeat victims—i.e., businesses that suffered two or more extortion incidents in 2013— constituted 2% of all respondents (27% of victims) but accounted for 56% of all extortion incidents. Businesses that experienced three or more incidents amounted to less than 1% of the sample (12% of victims), yet suffered 38% of all incidents of extortion. Moreover, the distribution clearly exhibits overdispersion (Cameron and Trivedi 2013, p. 4), with its variance ( 0.547 ) being more than four times larger than its mean ( 0.133). 

It is also relevant to note that the tail of the distribution in Table 1 exhibits a clustering of responses at 10 events, with responses for greater counts occurring only at multiples of 5 or 6. As (Farrell and Pease 2007, p. 45) note, such clustering is ubiquitous in victimization studies and reflects respondents’ tendency to estimate high frequency event counts around likely reference points (such as multiples of 5 or 10 for the decimal system, or of 6 and 12 for the calendar year). It is generally assumed that some victims are victimized so often that they must use heuristics to estimate the amount of victimization incidents experienced. This reflects the well-known just noticeable difference phenomenon (Farrell and Pease 2007 2015 ), also known as difference threshold (Colman ). While this effect does introduce some potential measurement error, it is important to note that such victim estimation 

> 13 Available upon request. 

> 14 The ENVE defines extortion as “any kind of threat or coercion committed against the local unit’s owner or staff for the purpose of obtaining money, goods or forcing them to do or stop doing something” (Jaimes Bello and Vielma Orozco 2013, p. 172). This definition is similar to that adopted by Chin et al. (1992) in a victimization survey of businesses in New York “Chinatowns,” insofar as it treats “demanding money or the provision of goods and services to avoid violence or harassment” as the working definition of extortion (p. 629). Given that only licit—i.e. officially registered—businesses were sampled, this study does not consider extortions committed against informal—i.e. unregistered—businesses, or those against other criminal actors. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1127 

does not discredit victimization measurements (see Farrell and Pease 2007, p. 40–46, for a discussion). 

Alternatively, such clustering may be explained by institutionalized racketeering, as is expected in the case of organised crime-related extortion. In such contexts, businesses could be expected to pay extortion demands at regular intervals (e.g. monthly, fortnightly, weekly), and would thus also cluster at said intervals. As INEGI (2014c, p. 29) notes, if a business is extorted weekly the survey would capture 52 incidents, a figure not much greater than the maximum event count observed in Table 1. 

Considering the state-level distribution of extortion (see panels A and B in Fig. 1), it is evident that extortion is not uniformly distributed across the country. Notably, the state distribution of the prevalence of extortion (panel A) is not the same as that of its concentration (panel B). This would suggest that states where becoming the victim of extortion is more likely are not necessarily the same where repeat extortion victimization is more frequent. To explore this further, we examined the bivariate relationship between state-level prevalence and concentration (see Fig. 2). The analysis suggested that there was no statistically significant association between extortion prevalence and concentration at the state level ( _R_ = −0.086 , _p_ = 0.64 ), which supports the view that prevalence and extortion are fueled by distinct processes (at least at the state level).<sup>15</sup> 

### **Independent Variables** 

Given the potential predictors identified in an earlier section, four macro-level variables measured at the state level ( _corruption prevalence (log)_ , _federal weapon crimes (log)_ , _federal drug crimes (log)_ , and _rule of law_ ), and four micro-level variables measured at the level of individual business units ( _corruption victimizations_ , _years in business_ , _business type_ , and _business size_ ) were operationalized as independent variables. We included three additional macro-level variables ( _number of surveyed businesses (log)_ , _population (log)_ and _competitiveness_<sup>16</sup> ) as controls. Table 2 presents descriptive statistics for the independent variables, while Fig. 1 presents a series of thematic maps summarizing how these variables, as well as key indicators of extortion, vary across Mexico. 

Macro-level variables 

- _Rule of law_ : To measure the strength of the legitimate governance structure, we used a rule of law index obtained from the Mexican Institute for Competitiveness ( _Instituto Mexicano para la Competitividad_ , IMCO 2016), a think tank.<sup>17</sup> 

- _Corruption prevalence_ : Measures the number of businesses in a state that reported being the victim of corruption in the ENVE. 

> 15 In contrast, see (Sidebottom 2013, p. 154–155) for a positive and statistically significant relationship between area-level prevalence and concentration of burglary victimization in Malawi, which suggests that at least some of the area-level factors affecting burglary victimization in that country also affect the number of repeat burglaries suffered. 

> 16 We used a modified version of IMCO’s _competitiveness_ index that assesses states on a 100 point scale (higher is better) according to their 2013 performance in nine subindices measuring business friendliness. The version we used excluded the rule of law component as this is used as a dependent variable. 

> 17 We used a revised version of the index grading states on a 100 point scale (higher is better) based on 2013 measures of kidnapping incidence, vehicle theft, costs of crime, total personal and household crime incidence, the dark figure, fear of crime, availability of notaries, and contract enforcement. We excluded homicide rates from the revised index, as these were collinear with our other independent variables. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1128 

**Table 1** The distribution of extortion victimization and the percentage of potential targets affected 

|Events|Prevalence|Incidence|Target %|Victim %|Incident %|Repeats %|
|---|---|---|---|---|---|---|
|0|25895|–|91.953|–|–|–|
|1|1654|1654|5.873|72.992|44.236|–|
|2|338|676|1.200|14.916|18.080|22.946|
|3|139|417|0.494|6.134|11.153|18.873|
|4|55|220|0.195|2.427|5.884|11.202|
|5|22|110|0.078|0.971|2.942|5.974|
|6|12|72|0.043|0.530|1.926|4.073|
|7|3|21|0.011|0.132|0.562|1.222|
|8|8|64|0.028|0.353|1.712|3.802|
|10|20|200|0.071|0.883|5.349|12.220|
|12|3|36|0.011|0.132|0.963|2.240|
|15|4|60|0.014|0.177|1.605|3.802|
|20|3|60|0.011|0.132|1.605|3.870|
|24|1|24|0.004|0.044|0.642|1.561|
|25|1|25|0.004|0.044|0.669|1.629|
|30|2|60|0.007|0.088|1.605|3.938|
|40|1|40|0.004|0.044|1.070|2.648|
|Totals|28161|3739|100%|100%|100%|100%|



- _Federal drug crimes_ : To estimate the amount of drug trafficking activities in each state, we used the number of crimes per state related to the General Health Law (used to reg- _Pro-_ 

- ulate prohibited substances) as reported in 2013 by the Attorney General’s Office ( _curaduría General de la República_ , PGR) to the Executive Secretariat of the National System for Public Security ( _Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública_ , SESNSP 2015). 

- _Federal weapon crimes_ : The number of crimes relating to the Federal Law on Firearms and Explosives as reported for 2013 by the PGR to the SESNSP (2015). This variable serves as a proxy measurement of organized crime groups’ capacity to inflict violence, as such crimes refer to those involving high-powered weapons—as well as seizures of such weapons and ammunition—traditionally associated with organized crime groups. 

### Micro-level variables 

- _Corruption victimizations:_ Captured as counts in the ENVE by the question: “In total, how many separate acts of corruption did you suffer during 2013?” (INEGI 2014d). An act of corruption refers to a situation where a public servant—or a third party acting on their behalf—directly asked for, suggested, or set the conditions for the payment of a bribe by the business (Jaimes Bello and Vielma Orozco 2013; INEGI 2014d). 

- _Years in business:_ Calculated by subtracting the year respondents reported that their business started operations from the survey reference year (i.e. 2013). As our interest is modeling the effect of being a new business in comparison to older businesses, rather than the effect of an additional year in business, nominal categories were considered more appropriate. Thus, businesses were binned into quintiles from the 20% youngest to the 20% oldest. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1129 

**Table 2** Descriptive statistics of independent variables 

|Variable|Obs.|%|Mean|S.D.|Min.|Max.|
|---|---|---|---|---|---|---|
|_Business-level variables_|||||||
|Corruption victimizations|28161||0.1|1.3|0|98|
|_Years in business_|||||||
|0–5 (_base_)|6772|24.0|||||
|6–9|5921|21.0|||||
|10–14|4984|17.7|||||
|15–23|5500|19.5|||||
|24–212|4984|17.7|||||
|_Business type_|||||||
|Retail (_base_)|10088|35.8|||||
|Mining|89|0.3|||||
|Construction|820|2.9|||||
|Manufacturing|3707|13.2|||||
|Wholesale|1952|6.9|||||
|Transport|720|2.6|||||
|Media|259|0.9|||||
|Finance|318|1.1|||||
|Real estate|417|1.5|||||
|Prof. services|753|2.7|||||
|Maintenance|908|3.2|||||
|Education|955|3.4|||||
|Health|1157|4.1|||||
|Leisure|316|1.1|||||
|Hotels, Rest. & Bars|2787|9.9|||||
|Other|2915|10.3|||||
|_Size_|||||||
|Large (_base_)|3052|10.8|||||
|Medium|3640|12.9|||||
|Small|5840|20.7|||||
|Micro|15629|55.5|||||
|_State-level variables_|||||||
|Corruption prevalence|32||40.1|18.4|14|101|
|Federal weapon crimes|32||559.6|451.6|31|1632|
|<br>Federal drug crimes|32||526.9|871.7|37|3738|
|Rule of law index|32||54.4|13.3|21.4|78.4|
|Competitiveness index|32||47.7|8.5|25.3|67.8|
|Population (in millions)|32||3.7|3.15|0.7|16.4|
|N sampled businesses|32||880.6|236.4|534|1657|



1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1130 



<!-- Start of picture text -->
Mexico: State−level summary statistics<br>A. Extortion prevalence B. Extortion concentration C. Corruption prevalence<br>D. N surveyed businesses E. Federal Drug Crimes F. Federal Weapon Crimes<br>s.d.<br>4<br>2<br>0<br>−2<br>G. Population H. Business competitiveness index I. Rule of law index<br>N<br>500 km 1000 km<br>Geometries: INEGI, ESRI; Data: ENVE 2014, SESNSP, IMCO<br><!-- End of picture text -->

**Fig. 1** Thematic maps showing variations in state-level variables, and state-level measures of extortion 

**Fig. 2** Relationship between state-level extortion prevalence and concentration. The results of a correlation test (shown in label) suggest that there was no statistically significant relationship between the variables 



<!-- Start of picture text -->
2.5<br>R = -0.086, p = 0.64<br>2.0<br>1.5<br>50 100 150<br>Prevalence per 1000 businesses<br>Concentration<br><!-- End of picture text -->

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1131 

- _Business type:_ Captured according to the North American Industrial Classification System’s second level<sup>18</sup> ( _Sistema de Clasificación Industrial de Norte América_ , SCIAN, INEGI 2007). 

- _Business size:_ Categories (micro, small, medium and large) were provided by INEGI (2014b) and are based on the number of employees reported by each business.<sup>19</sup> 

### **Analytical Strategy** 

Our analysis was conducted in two parts. First, using an implementation of the Kolmogorov–Smirnov test (KS test, Upton and Cook 2014a) for discrete distributions proposed by Arnold and Emerson (2011), we assessed H1 by comparing the observed distribution of extortion to that expected under the null hypothesis of random victimization—the latter estimated by a Monte Carlo simulation of a Poisson process with 500 replicates. Additionally, to assess if the observed distribution presented more zeroes than expected, we compared the observed prevalence to that expected under simulated Poisson and negative binomial distributions using contingency tables. 

Next, we used statistical modeling to assess H2. The standard model used to estimate victimization counts is the multilevel negative binomial<sup>20</sup> model (MNB) (Tseloni and Farrell 2002). Considering _yij_ is the count of extortion victimizations suffered by the _i_ th business, in the _j_ th state, and that _xij_<sup>′is a vector of covariates thought to determine</sup><sup>_yij_, the MNB</sup> model for the mean of event counts, E[ _yij_ | _xij_<sup>�] =</sup><sup>_휇ij_, can be represented by:</sup> 



> where _훽_ 0 is the intercept, and _훽_ 1 is a vector of fixed regression coefficients that quantify the relationship between _xij_<sup>′and ln(</sup><sup>_휇ij_) .</sup><sup>_u_0</sup><sup>_j_is the random variation in</sup><sup>_훽_0 associated with each</sup> state _j_ , and _휀ij_ is a gamma distributed error term that incorporates overdispersion via the _훼_ parameter (see Cameron and Trivedi 2013; Tseloni and Farrell 2002; Hilbe 2011). The probability function of the MNB model (see Cameron and Trivedi 2013; Hilbe 2011, ) is given by: 



18 The categories are, in industry: mining, construction, and manufacturing; in commerce: retail and wholesale; in services: transport, media, finance and insurance, real estate, professional scientific and technical services, maintenance providers, education, health, leisure, restaurants, hotels and bars, and other services. Observations corresponding to 18 businesses classified as utilities and corporate offices were excluded as there were problems of complete and quasi-complete separation. 

> 19 Micro businesses have 10 employees or fewer; small businesses employ between 11 and 50 people (11– 30 in commerce); medium businesses employ between 51 and 250 in industry, 31–100 in commerce, and 51 to 100 in services; large businesses are those with 101 or more employees (251 or more in industry). 

> 20 We used the standard NB2 formulation of the negative binomial variance function, _휎ij_ 2<sup>=</sup><sup>_휇ij_+</sup><sup>_훼휇_</sup> _ij_<sup>2 (Cam-</sup> eron and Trivedi 2013). 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1132 

When _훼_ → 0 , the probability function collapses to the Poisson (Cameron and Trivedi 2013, p. 85), meaning that there is no business-level unexplained heterogeneity. Conversely as _훼_ becomes larger, unexplained heterogeneity between businesses increases. Heterogeneity at the state level is captured by _휎u_<sup>2</sup> 0<sup>, and the formula</sup><sup>_휌_=</sup><sup>_휎_</sup> _u_<sup>2</sup> 0<sup>∕(</sup><sup>_휎_</sup> _u_<sup>2</sup> 0<sup>+</sup><sup>_훼_) , allows the calcu-</sup> lation of the intra-class correlation (ICC), which represents the correlation of the mean of extortion incidents between two identical businesses in the same state (see Goldstein 2011; Tseloni and Farrell 2002). Conversely, the inverse of the ICC, 1 − _휌_ , represents the probability of two identical businesses anywhere in the country experiencing the same number of extortion victimizations, after controlling for between-state differences. 

We hypothesized (H2) that extortion concentration is produced by a process different from that which generates extortion prevalence, thus the MNB model is unsuitable for our purposes—as it uses the same probability function (Eq. 2) for all values of _y_ . The multilevel negative binomial-logit hurdle model (MNB-LH), in contrast, is a suitable alternative. The fundamental logic underpinning hurdle models (Mullahy 1986) is that they allow the specification of distinct probability functions for observations where _y_ = 0 , and _y >_ 0 (Cameron and Trivedi 2013): 



where _f_ 1(0) is the probability of observing a zero count. If the hurdle is crossed (if _y >_ 0 ), the truncated count density is given by _f_ 2( _y_ )∕(1 − _f_ 2(0)) , which needs to be multiplied by 1 − _f_ 1(0) to ensure that probabilities sum to one (Cameron and Trivedi 2013). In practice, ⋅ _f_ 1( ) can be estimated using the logit model, specified in its multilevel form (Goldstein 2011) as such: 



where _훾_ 0 is the intercept for the binary model, _훾_ 1 is a vector of fixed coefficients that quantify the relationship between _xij_<sup>′and ln(</sup><sup>_휋ij_) , and</sup><sup>_v_0</sup><sup>_j_represents the random variation in</sup><sup>_훾_0</sup> associated with each state _j_ . The probability of observing zero ( _f_ 1(0) ) is given by (Hilbe 2011): 



and the probability of crossing the hurdle ( 1 − _f_ 1(0) ) is thus (Hilbe 2011): 



Taking the standard negative binomial density in Eq. 2 as _f_ 2(⋅) , the truncated density needs to be rescaled as shown in Eq. 3. The _f_ 2(0) density is (1 − _훼휇ij_ )<sup>−1∕</sup><sup>_훼_</sup> (Hilbe 2011), and 1 − _f_ 1(0) is shown in Eq. 6. Thus, the multilevel truncated negative binomial density for _yij >_ 0 is: 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1133 



where _휇ij_ is estimated using Eq. 1 and restricting _yij >_ 0 . The complete multilevel negative binomial-logit hurdle model is therefore: 



An advantage of the hurdle model is that its components (the prevalence and truncated count models) can be estimated separately. However, when observations are clustered and random effects are used, as in the multilevel approach used here, the two stage estimation procedure assumes that group random effects for prevalence and concentration ( _v_ 0 _j_ and _u_ 0 _j_ , respectively) are independent, when in fact they may be correlated (Min and Agresti 2005). This situation would arise when unobserved differences between states affect state-level prevalence and concentration in the same way, and could bias the results if not addressed. In response, Min and Agresti (2005) suggest fitting the hurdle model with random effects following a bivariate normal distribution and estimating the model via approximations (see also Cantoni et al. 2017). 

Unfortunately, the software available at INEGI did not permit the fitting of a hurdle model with bivariate normal random effects. While this could potentially lead to some bias in the estimates, (Cantoni et al. 2017, p. 2191) note that estimated coefficients and standard errors are generally robust to this misspecification (see also McCulloch and Neuhaus 2011). Furthermore, Cantoni et al. (2017) note that when there is no correlation between cluster prevalence and concentration, estimates from independently estimated hurdle models are unbiased. As discussed in the “Dependent Variable” section, we found no relationship between prevalence and concentration of extortion at the state level, thus the assumption of correlated random effectcs can be relaxed and the independent hurdle approach used can be judged as appropriate. 

All models were estimated using the “glmmADMB” package (Fournier et al. 2012; Bolker et al. 2012). We estimated the standard MNB model as a baseline to compare the estimates of the MNB-LH model. The MNB-LH model was estimated separately, with a multilevel logit (ML) estimating the likelihood of observing a victimization incident, and a multilevel truncated negative binomial (MTNB) estimating the expected concentration among victimized targets. Model significance of each individual model (MNB, ML and MTNB) was assessed using likelihood ratio tests (Cameron and Trivedi 2013, p. 49; Hilbe 2011, p. 177). Given that MNB and MNB-LH models are not nested—and hence not suitable for comparison using likelihood ratio tests— they were compared using the Akaike (AIC) and Bayesian (BIC) information criteria (Cameron and Trivedi 2013, p. 197), with lower values indicating a better model. AIC and BIC estimates for the hurdle model were calculated by adding the AIC and BIC estimates of the constituent models (Hilbe 2014, e.g. _AICMNB_ - _LH_ = _AICML_ + _AICMTNB_ ; see][p. 188). 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1134 

## **Results** 

### **Univariate Analysis** 

Table 1 shows the distribution of crimes across targets. This can also be displayed using Lorenz curves (Upton and Cook 2014b), which show the cumulative share of crime experienced by the cumulative share of the population that experiences them (Tseloni and Pease 2005). If crime is evenly distributed, the Lorenz curve will approximate the line of equality shown in the figure. Deviation from this indicates that crime is unequally distributed (Tseloni and Pease 2005, p. 77). 

Figure 3 shows Lorenz curves for the observed and expected distributions, the latter being the distribution of the simulated replicates. The panel on the left shows that both observed and expected frequencies are highly concentrated among all businesses, which is to be expected given than the vast majority of businesses were not extorted. The right-hand panel, however, shows the distribution for victimized businesses—thus representing repeat victimization. The curve of the expected distribution is very close to the line of equality, while the curve of the observed distribution exhibits far more concentration. Crucially, a KS test ( D = 0.044 , _p <_ 0.001 ) confirmed that the differences between the distributions are statistically significant—there is more repeat victimization than that expected under random victimization. 

Lastly, considering the shape of the statistical distribution, Table 3 and Fig. 4 show that there are more zeros that would be expected assuming a Poisson distribution ( _휒_<sup>2</sup> = 498.2 , df = 1, _p <_ 0.001 ). However, after taking account of overdispersion by modeling a negative binomial distribution, there were no significant differences between the number of observed and expected zeroes ( _휒_<sup>2</sup> = 0.17 , df = 1, _p_ = 0.68 ), hence the modeling strategy does not need to account for zero-inflation. 

### **Statistical Modeling** 

Table 4 presents model statistics for null and fully specified versions of estimated models. Goodness of fit was assessed using likelihood ratio tests. Fully specified models were found to be significantly different from null models. Multilevel specifications significantly improved fit when compared to single-level models. Similarly, the MNB and MTNB models proved a significant improvement over Poisson and truncated Poisson models. AIC and BIC values for the hurdle model were smaller than for the MNB model ( _AICMNB_ − _LH_ − _AICMNB_ = −219 , and _BICMNB_ − _LH_ − _BICMNB_ = −38 ), which suggests that the hurdle model of extortion victimization is more appropriate. The table also presents estimates for state-level variance, the _훼_ parameter, and the intra-class correlation (ICC), which are discussed in detail in a later section. 

Lastly, multi-collinearity was not deemed to be significant, as generalized variance inflation factors (Fox and Monette 1992, GVIF,][) were quite small (see Table 5). As (Fox and Monette 1992 , p. 180) note, to preserve comparability inflation factors must be scaled by the amount of categories in categorical predictors. Adjusting for this, the largest value in column GVIF<sup>1∕2</sup><sup>_df_</sup> is 2.08—well within the thresholds that many practitioners regard as a sign of severe multi-collinearity<sup>21</sup> (O’brien 2007). 

> 21 See also the online discussion on the Cross Validated website: https ://bit.ly/3bt6f 98 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1135 

**Fig. 3** Lorenz curves with the observed and expected distributions of extortion victimization 



Figure 5 shows exponentiated model coefficients<sup>22</sup> to facilitate interpretation ( _e_<sup>_훽_</sup> and _e_<sup>_훾_</sup> ). For count MNB and MTNB models, the exponentiated estimates represent incidence rate ratios (IRR, Hilbe 2014, p. 60), whereas for the binary ML model they represent odds ratios (OR, Weisburd and Britt 2014, p. 568). Subtracting 1 from the IRR ( _IRR_ − 1 ) gives the percentage change on the concentration of extortion victimization for a one unit increase in the independent variable, while _OR_ − 1 gives the percentage change in the prevalence risk. For categorical independent variables, the percentage change is relative to the reference category. IRRs and ORs for log transformed independent variables<sup>23</sup> represent change for a 10% increase in the independent variable (given by 1.10<sup>_훽_</sup> and 1.10<sup>_훾_</sup> ). 

### **Macro‑level Effects** 

Overall, the associations between extortion and state-level variables were quite weak. Moreover, the associations were inconsistent for the prevalence and concentration components, as no state-level variable was significant in the count part (MTNB) of the hurdle model. Thus, the associations in the MNB model appear to reflect differences in prevalence risk captured by the ML model, rather than a state-level effect on repeat extortion victimization. 

All else being equal, the ML model found that a 10% increase in the number of corruption victims in a state was associated with a 5% increase in the likelihood of a business becoming a victim of extortion. Similarly, a 10% increase in the number of federal weapon crimes in a state, was associated with a 4% increase in extortion prevalence risks. In contrast, a 10% increase in the number of federal drug crimes in a state was associated with a 2% reduction in the likelihood of a business becoming a victim of extortion. Lastly, differences in the rule of law index between states showed no association with extortion risks. 

> 22 Raw model estimates can be found in the “Appendix”. 

> 23 Log transformed variables were centered around the log of the mean. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1136 

|**Table 3**Observed and expected<br>prevalence calculated using a||Observed|Poisson|Neg. Bin.|
|---|---|---|---|---|
|Monte Carlo simulation with<br>2000 replicates|0|25895|24659|25914|
||≥1|2266|3502|2247|
||_Total_|28161|28161|28161|





<!-- Start of picture text -->
0.0075<br>0.0050 Distribution<br>Poisson<br>Negbin<br>0.0025<br>0.0000<br>24500 25000 25500 26000<br>Zeroes<br>density<br><!-- End of picture text -->

**Fig. 4** Amount of zeroes predicted by 2000 Monte Carlo replicates of a Poisson and a Negative Binomial distribution based on the observed distribution of extortion. The observed prevalence of zeroes is within the 95% CI of the negative binomial distribution but outside the Poisson expectation 

**Table 4** Model statistics for null and fully specified multilevel negative binomial (MNB), multilevel negative binomial-logit hurdle (MNB-LH), multilevel logit (ML) and multilevel truncated negative binomial (MTNB) models 

||MNB||MNB-LH||ML||MTNB||
|---|---|---|---|---|---|---|---|---|
||Null|Full|Null|Full|Null|Full|Null|Full|
|Log. lik.|− 10111|− 9841|− 9992|− 9700|− 7679|− 7455|− 2313|− 2244|
|AIC|20228|19748|19995|19529|15363|14975|4632|4555|
|BIC|20253|20020|20029|19982|15379|15239|4649|4744|
|_휎_<sup>2</sup>|0.23|0.08|–|–|0.25|0.09|0.31|0.21|
|_훼_|9.12|7.48|–|–|–|–|148.41|148.41|
|ICC|0.02|0.01|–|–|–|–|0.00|0.00|
|LRT||540.3***||–||447.9***||137.7***|
|_n_|28161|28161|–|–|28161|28161|2266|2266|
|Groups|32|32|–|–|32|32|32|32|



Degrees of freedom for likelihood ratio tests (LRT), 30.<sup>∗∗∗</sup> _p <_ 0.001 

### **Micro‑level Effects** 

Business-level effects were also inconsistently associated across the components of the hurdle model, with the exception of corruption victimizations and being a micro-sized business. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1137 

**Table 5** Generalized variance inflation factors for independent variables 

||MNB and<br>|ML (_y_≥<br>|0)<br>|MTNB (_y_<br>|= 0)<br>||
|---|---|---|---|---|---|---|
||GVIF|df|GVIF<sup>1∕2</sup><sup>_df_</sup>|GVIF|df|GVIF<sup>1∕2</sup><sup>_df_</sup>|
|Corruption victimizations|1.01|1|1.00|1.02|1|1.01|
|Years in business|1.12|4|1.01|1.14|4|1.02|
|Business type|1.38|15|1.01|1.43|15|1.01|
|Size|1.46|3|1.07|1.36|3|1.05|
|Corruption prevalence (log)|1.81|1|1.35|1.94|1|1.39|
|Weapon crimes (log)|3.69|1|1.92|4.32|1|2.08|
|Drug crimes (log)|2.62|1|1.62|2.66|1|1.63|
|N businesses (log)|1.82|1|1.35|2.12|1|1.46|
|Population (log)|2.15|1|1.47|2.23|1|1.49|
|Competitiveness index|1.23|1|1.11|1.23|1|1.11|
|’Rule of law’ index|1.72|1|1.31|2.27|1|1.51|



The MNB model indicates that a one unit increase in the number of corruption victimizations experienced by a business increases the number of extortion victimizations a business can expect by 32%. The hurdle components suggest that the effect of corruption victimizations is consistent for both the prevalence and concentration of extortion. The ML model suggests that a one unit increase in the number of corruption victimizations increases the likelihood of becoming a victim of extortion by 12%, whereas the MTNB model suggests that it increases the number of repeat extortions expected by extorted businesses by 15%. 

Business size categories had contrasting effects for the prevalence and concentration components. Results from the MNB model suggest that micro-sized businesses suffer an average of 50% fewer extortion incidents than large businesses (the reference category). The insignificant coefficients for small and medium-sized businesses in the MNB model suggest that they suffer extortion victimization at the same rate as large businesses. However, the results of the hurdle model paint a more nuanced picture. The effect of being a micro-sized business is consistent across the hurdle: they are 30% less likely to become victims of extortion, and experience 66% fewer extortion repeats if victimized. On the other hand, small businesses face a 42% higher risk of becoming a victim of extortion, yet they experience 52% fewer extortion repeats once extorted. Medium businesses also see higher risks of extortion prevalence (23%), yet they experience repeat extortion at the same rate as large businesses, as the MTNB coefficient was not significant. 

With few exceptions, most business types faced no difference in extortion risks relative to retailers (the reference category), though some categories presented contrasting effects across the hurdle components. Hotels, restaurant and bars suffer 48% more extortion incidents than retailers (MNB), though the hurdle model suggested that this was mainly due to differences in the prevalence risk, rather than due to repeat victimization. The category faced a 43% higher risk of becoming a victim of extortion (ML), though it showed no significant effect in the concentration of repeat victimization (MTNB). Manufacturers, maintenance service providers, and media businesses experienced fewer extortion incidents overall (from MNB: −19 %, −27 %, and −78 %, respectively), though for manufacturers and media businesses, such 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1138 

|●<br>1.15***<br>●<br>1.09<br>●<br>1.15<br>●<br>1.19<br>●<br>1.36<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>0.75<br>0.67<br>0.87<br>1.05<br>0.49**<br>0.24<br>0.78<br>0.97<br>0.87<br>0.4**<br>0.79<br>1.05<br>0.96<br>1.35<br>1.08<br>0.78<br>0.48***<br>0.34***<br>1.01<br>1.01<br>0.97<br>1.03<br>0.97<br>0.98<br>1.01<br>_Hurdle_: Multilevel Truncated Negative Binomial (MTNB)<br>0<br>1<br>2<br>3<br>4|
|---|
|●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>1.12***<br>1.39***<br>1.43***<br>1.56***<br>1.42***<br>0.87<br>1.08<br>0.84*<br>1.08<br>1.26<br>0.3**<br>1.16<br>1.3<br>1.19<br>0.96<br>0.87<br>1.11<br>0.89<br>1.43***<br>0.86<br>1.23*<br>1.42***<br>0.7***<br>0.99<br>1.04***<br>1.05**<br>0.95<br>0.98**<br>0.98*<br>1<br>_Hurdle_: Multilevel Logit (ML)<br>0.5<br>1.0<br>1.5<br>Exponentiated estimates and 95% C.I.<br>∗_p <_0.001,∗∗_p <_0.01,∗_p <_0.05|
|●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>1.22<br>●<br>1.2<br>●<br>1.26<br>●<br>0.73*<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>●<br>1.32***<br>1.34***<br>1.5***<br>1.54***<br>1.55***<br>0.72<br>0.98<br>0.81*<br>1.07<br>0.97<br>0.22***<br>0.86<br>1.15<br>0.81<br>1.48***<br>0.86<br>1.12<br>1.02<br>0.5***<br>1.03*<br>1.04***<br>0.97***<br>0.96<br>0.99<br>0.98**<br>1<br>Multilevel Negative Binomial (MNB)<br>0.5<br>1.0<br>1.5<br>2.0<br>State: 'Rule of law' index<br>State: Competitiveness index<br>State: Population (log)<br>State: N businesses (log)<br>State: Drug crimes (log)<br>State: Weapon crimes (log)<br>State: Corruption prevalence (log)<br>Size: Micro<br>Size: Small<br>Size: Medium<br>Type: Other<br>Type: Hotels Rest Bar<br>Type: Leisure<br>Type: Health<br>Type: Education<br>Type: Maintenance<br>Type: Prof. services<br>Type: Real estate<br>Type: Finance<br>Type: Media<br>Type: Transport<br>Type: Wholesale<br>Type: Manufacturing<br>Type: Construction<br>Type: Mining<br>Years: 24 to 212<br>Years: 15 to 23<br>Years: 10 to 14<br>Years: 6 to 9<br>Corruption victimizations<br>**. 5**Forest plot with exponentiated results from the models. Signifcance:∗∗|



1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1139 

−16 % and differences were apparently due to a lower risk of becoming victims (from ML: −70 % respectively), as they did not suffer differential rates of repeat extortion according to MTNB estimates. On the other hand, the opposite appears to be true for maintenance service providers, whose odd ratios were not significantly different from 1, though they experienced 60% less repeat extortion than retailers. Lastly, while overall MNB and ML estimates did not reveal significant differences between retailers and transport providers, MTNB estimates suggest that once victimized, transport providers experienced 51% fewer repeats. 

Business age also showed a contrasting effect across the components of the hurdle model. Though the MNB model showed a positive and significant effect for all age categories, the hurdle model suggests that this is due to changes in prevalence risk. According to estimates from the ML model, businesses aged 6–9, 10–14, 15–23, and 24 years or more are 39%, 43%, 56%, and 42% respectively more likely to become victims of extortion, when compared with businesses that had been in operation for 5 years or fewer (the reference category). However, once victimized, business age categories had no effect on the expected concentration of repeat extortion, as the coefficients in MTNB were not significant. 

### **Unobserved Heterogeneity** 

Unobserved heterogeneity refers to differences in extortion victimization that remain unexplained by the independent variables in the study. Between-business unobserved heterogeneity ( _훼_ in Table 4) arises when two identical businesses—in terms of those characteristics included in the model—suffer unexplained differences in extortion incidence. The independent variables selected reduced individual unobserved heterogeneity by 18%, from _훼null_ = 9.12 in MNB, to _훼_ = 7.48 in the fully specified MNB model. The relatively high value of unexplained heterogeneity that remains suggests that there are factors not included in the model that clearly influence extortion concentration. Such variables could relate to risk heterogeneity or event dependence, though it is not possible to tell with this model. On the other hand, the very substantive amount of unobserved heterogeneity in the count part of the hurdle model ( _훼_ = 148.41 in MTNB), which was essentially unaffected by the inclusion of explanatory variables associated with risk heterogeneity, suggests that an alternative process, such as event dependence, may be responsible. 

In contrast, unobserved heterogeneity between states ( _휎_<sup>2</sup> in Table 4) refers to differences in extortion risk faced by businesses in different states, after controlling for the variables specified in the model. In the MNB model, level 2 variance is reduced significantly by the inclusion of independent variables ( model, suggests that much of this reduction is due to differences in the risk of extortion −65 %, from _휎u_<sup>2</sup> 0 _null_<sup>= 0.23 to</sup><sup>_휎_</sup> _u_<sup>2</sup> 0<sup>= 0.08 ). The hurdle</sup> prevalence, rather than in repeat victimization, as the ML models saw reductions of 64% (from _휎u_<sup>2</sup> 0 _null_<sup>= 0.25 to</sup><sup>_휎_</sup> _u_<sup>2</sup> 0<sup>= 0.09 ) and the MTNB models reduced between-states unob-</sup> served heterogeneity by 32% ( _휎u_<sup>2</sup> 0 _null_<sup>= 0.31 to</sup><sup>_휎_</sup> _u_<sup>2</sup> 0<sup>= 0.21).</sup> 

Both the MNB and the MTNB models suggest that the intra-state correlations (ICC in Table 4) are very small, and that between-businesses variations are more relevant. When comparing all businesses, the ICC for MNB suggests that the probability of two identical businesses—in terms of the variables included in the model—experiencing the same extortion incidence due to being in the same state was only 1%. Conversely, the probability of two identical businesses experiencing the same number of extortion victimizations, after controlling for between-state differences, was 99% ( 1 − _ICC_ ). For repeat victimization, the MTNB model suggests that this probability was 99.9%. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1140 



<!-- Start of picture text -->
4<br>3<br>2<br>1<br>Hurdle: Multilevel Truncated Negative Binomial (MTNB) 0<br>2.0<br>1.5<br>Excluding outliers<br>1.0<br>Bootstrap<br>Hurdle: Multilevel Logit (ML)<br>0.5 Original<br>Exponentiated estimates and 95% C.I.<br>2.0<br>1.5<br>1.0<br>Multilevel Negative Binomial (MNB) 0.5<br>Corruption victimizations Years: 6 to 9 Years: 10 to 14 Years: 15 to 23 Years: 24 to 212 Type: Mining Type: Construction Type: Manufacturing Type: Wholesale Type: Transport Type: Media Type: Finance Type: Real estate Type: Prof. services Type: Maintenance Type: Education Type: Health Type: Leisure Type: Hotels Rest Bar Type: Other Size: Medium Size: Small Size: Micro State: Corruption prevalence (log) State: Weapon crimes (log) State: Drug crimes (log) State: N businesses (log) State: Population (log) State: Competitiveness index State: 'Rule of law' index<br><!-- End of picture text -->

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1141 

## **Robustness Checks** 

We performed a series of post hoc analyses to check the robustness of our results. First we calculated bootstrap standard errors for all models. To do this, we randomly sampled with replacement _n_ business units 99 times, fitted models to each sample, and calculated the standard error of the bootstrap coefficient estimates. Then, we fitted a further round of models to a subset of data excluding observations with 30 or more extortions, to rule out 6 a potential outlier effect. Figure presents a graphical comparison of coefficient estimates and confidence intervals for the original models and post hoc robustness checks. Complete results tables are available in the “Appendix”. 

Post hoc estimates were remarkably consistent with those of the original model. For all models, bootstrap standard errors for business-level variables were only marginally different from the original models, with no changes in inference. Bootstrap standard errors for state-level variables were consistently smaller, though the changes only affected inference for control variables, with the exception of drug crimes in the MTNB model. Similarly, the magnitude and significance of parameters from models excluding outliers were very similar to those of the original model, with no changes in inference, except for the coefficient for drug crimes in the MTNB model. 

Post hoc checks for unobserved heterogeneity were similarly consistent with original estimates. In the MNB model, business-level unobserved heterogeneity ( _훼_ ) decreased a marginal amount after excluding outliers, while it remained virtually unchanged in the MTNB model. Similarly, state-level heterogeneity ( _휎_<sup>2</sup> ) was largely unaffected after excluding outliers in all models. 

## **Discussion** 

This study set out to systematically examine victimization patterns of extortion against businesses in Mexico to determine if incidents concentrate on repeat victims, and to explore if the factors that explain the risk of becoming a victim of extortion also explain the concentration of repeat extortion victimization. 

Using data from Mexico’s commercial victimization survey, extortion incidents were found to concentrate above what would be expected by chance (H1), with repeat extortion victims suffering a disproportionate amount of total crime incidents—patterns consistent with findings on repeat victimization for most crime types in many countries (e.g. Farrell and Pease 2011; Farrell et al. 2005). In all, there were more repeat extortion victims than one-time victims, and close to 40% of all incidents were repetitions. 

The literature on repeat victimization has traditionally considered that the factors that explain the prevalence of victimization also account for its concentration (e.g Pease and Tseloni 2014, p. 31). However, given that the role of event dependence may be more prominent in determining the risk of repeat victimization in the case of extortion, and that extortion often leads to enduring victim-offender relationships, we hypothesized that the factors that explain extortion concentration would be distinct from those that explain extortion prevalence (H2). 

Using a multilevel negative binomial-logit hurdle model, we found support for H2. Overall, only two independent variables (corruption victimization and being a micro-sized business) showed a consistent effect for both prevalence and concentration. Most variables 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1142 

showed inconsistent effects (e.g., a positive effect for prevalence but a not significant effect for concentration), though some presented contradictory ones (i.e., a positive effect for prevalence and a negative effect for concentration).<sup>24</sup> 

The literature on protection theory (e.g. Kleemans 2018) suggests that area-level effects would likely be more relevant for extortion risks as opposed to specific factors associated with individual businesses. Our findings mostly contradicted this, as state-level variables were only marginally relevant for predicting extortion prevalence (with the exception of drug crimes, which post hoc checks suggested were significantly associated with extortion concentration). Notably, the direction of the effects of the significant variables did fit with theoretical expectations: Businesses in states with more corruption (and hence with poorer governance), and with more weapon-related crimes (and hence with more violence-prone organized crime groups) experienced a higher risk of becoming victims of extortion, while businesses in states with more drug trafficking activity faced lower risks of becoming victims of extortion, and less concentration if victimized. Unobserved state-level heterogeneity for extortion prevalence was comparatively smaller than for extortion concentration, though the latter was dwarfed by the residual between-business unexplained heterogeneity. This suggests that any area-level explanations for repeat extortion are unlikely to be found at the state level, and instead may be explained by variables measured at sub-state level (e.g. municipality, city, neighborhood). 

Business-level effects proved to be far more important in explaining extortion risks, though they mostly affected extortion prevalence rather than concentration. We hypothesized that an association at the micro-level between corruption and extortion could be explained by direct relationships between extortionists and corrupt officials in Mexico— a link documented in the literature (Díaz-Cayeros et al. 2015; Morris 2013). While our analysis cannot refute the possibility of a spurious relationship (e.g. that extortionists and corrupt officials are attracted to the same kind of businesses), the fact that corruption victimization increases the likelihood of becoming a victim of extortion and the amount of repeat extortion incidents that businesses suffer once they have been extorted suggests that the relationship is substantive and robust. Nonetheless, having established that an association exists, exploring this issue further would seem to be important for future work. 

Business size categories showed inconsistent relationships for prevalence and concentration components—with the exception of micro sized businesses which were consistently less likely to be extorted, and suffered fewer repeats once victimized. Small businesses were more likely to be extorted than large businesses (possibly due to their relative vulnerability), though they suffered significantly fewer repeat victimizations thereafter (as potential rewards were possibly clarified following the first offense). Similarly, the higher 

> 24 Such inconsistencies of direction, magnitude and significance in the predictors may appear surprising, but they are in fact common findings in studies using hurdle models. For example, in a seminal study using a hurdle model to analyze healthcare utilization in Germany, Pohlmeier and Ulrich (1995) found that many of the predictors for the prevalence of visiting a physician were inconsistent with the predictors of the amount of visits by patients with at least one visit, indicating that prevalence and concentration of healthcare utilization were fueled by distinct processes. Similar inconsistencies across hurdle components can be found in studies of healthcare utilization in Mexico (Brown et al. 2005), workplace training (Arulampalam and Booth 1997), intimate partner violence (Hellemans et al. 2015), and sentencing decisions (Rydberg et al. 2017), for example. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1143 

prevalence risk for medium businesses suggests more inherent vulnerability, though the similar rates of repeat extortion suggest less variability regarding potential rewards. 

As expected, hotels, restaurants and bars did experience higher risks of extortion, though they did not experience more repeat incidents. The lower prevalence risks amongst manufacturers may point to accessibility as an additional factor influencing vulnerability to extortion: manufacturers tend to interact mostly with other businesses, while hotels, restaurants and bars are generally open to the public, and thus are naturally easier to access for extortionists. In contrast, the lower prevalence risk for media businesses may be related to risks perceived by extortionists, as extorting a news outlet such as a newspaper or a television station could lead to exposure. Results also showed that after being victimized once, most business types experienced similar rates of repeat extortion, apart from maintenance and transport service providers, which experienced less repeat extortion. We believe that such negative relationship may be related to factors affecting event dependence, such as victim response to the first offense (Farrell et al. 1995, p. 396). For example, transport providers may decide to avoid certain routes on which they have been previously victimized. 

Contrary to our expectations, new businesses (those with 5 or fewer years in operation) experienced substantially lower risks of becoming victims of extortion, though age was not associated with extortion concentration. We speculate that the association may be explained by target visibility, rather than by an inherent attractiveness or vulnerability linked to businesses’ age. All else being equal, new businesses may face lower risks because they are less likely be known by offenders—i.e. they are less likely to feature in 2011 an offender’s awareness space (Brantingham and Brantingham )—and “offenders can only commit crimes against targets of which they are aware” (Hepenstal and Johnson 2010 , p. 266). However, once a business is victimized (and thus known to offenders), businesses appear to be equally vulnerable and attractive to extortionists regardless of how long they have been in business. 

An important advantage of the hurdle model is that it allows clarifying the role of between-business unobserved heterogeneity for overall extortion risks (captured by the MNB model) and for the specific risks of extortion concentration (captured by the MTNB model). Many significant business-level predictors in the MNB model were in fact capturing differences in the prevalence risk, rather than in the risks of repeat extortion. Similarly, the between-business unobserved heterogeneity in MNB captures unexplained differences in both prevalence and concentration risks. By restricting observations to extortion victims, the between-business unobserved heterogeneity reported in the MTNB model refers only to unexplained differences in (repeat) extortion concentration. The high value of betweenbusiness unobserved heterogeneity, and the fact that it was unaffected by the inclusion of predictors, strongly support the hypothesis that (repeat) extortion concentration is fueled by a process distinct from that which explains extortion prevalence. We believe that this process is likely to be related to event dependence. As mentioned earlier, repeats are often associated with how victims respond to an initial event. Thus, if an initial extortion event were successful (in the offender’s view), it may lead to further extortion incidents in the future, as the victim is known to comply. This would be similar to the case of bank robbery, where success in past events has been found to be positively associated with suffering a repeat robbery (Matthews et al. 2001). On the other hand, in the case of extortion the relationship between past success and future risk could also be negative, as refusal to comply with an extortion attempt could lead repeated (and escalating) threats, especially if these occur in the context of an institutionalized extortion relationship—such as those observed in organized crime-related extortion rackets. However, we were unable to test whether the outcomes of previous events were associated with repeat extortion, as the data used in this 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1144 

study cannot clarify this. However, this certainly appears to be a relevant area for future research, should the necessary data become available. 

As discussed in section “A hurdle model of extortion victimization,” it is important to remember that model choice was primarily driven by study design: our main hypothesis required the use of a two-stage model that allowed us to test whether the predictors of the prevalence of extortion victimization were consistent with the predictors of its concentration. As Rose et al. (2006) discuss, choosing the “best” modeling approach when dealing with overdispersed count data should be based on a study’s aim and purpose, as models fit using zero-inflated and hurdle frameworks can sometimes be indistinguishable (see also Gray 2005). Thus, while alternative model frameworks (e.g. as zero-inflated models, or variations of the hurdle components used such as modeling concentration using a mixture of truncated counts) may also prove suitable to model extortion data, we did not consider their use to be within the scope of this study. This would certainly appear to be an important avenue for future research. 

There are, of course, other limitations to the present study. Chief amongst them is the validity of using victimization surveys to measure extortion against businesses. Di Gennaro and La Spina (2016; see also La Spina 2008; La Spina et al. 2014) suggest that victimization surveys are not reliable instruments to measure extortion (p. 4), though it is worth noting that their criticism is based on Italian victimization surveys, particularly one by Confcommercio-GFK Eurisko (2007), and another by TRANSCRIME (Mugellini 2012). Their main contention is that the surveys’ low response rates (6.3% and 14%, respectively) lead to self-selected samples unlikely to produce reliable outcomes (Di Gennaro and La Spina 2016, p. 4). Additionally, Asmundo and Lisciandra (2008) note that “victims of extortion are unlikely to come forward as such” in victimization surveys (p. 227). They ascribe this reticence to the fact that for many Italian—and especially Sicilian—businesses, “extortion is considered as ‘normal’ and made endogenous by the economic and social system, as an (ordinary) component of production costs,” and as such it loses its “criminal profile” (Asmundo and Lisciandra 2008, p. 227)—i.e. they do not consider themselves victimized. 

We find little evidence of these limitations in Mexico. First, the response rate for the ENVE’s random sample was much higher at 84% (Jaimes Bello and Vielma Orozco 2013), which should assuage fears of self-selection affecting measurement validity. Second, there is no evidence of a process of extortion “endogenisation” (Asmundo and Lisciandra 2008) in Mexico; in fact, the opposite appears to be true. Faced with increases in extortion and other organized crime related violence, the response from the business community in Mexico has ranged from vociferous protest, to proactive involvement in the improvement of public security institutions (Shirk et al. 2014). For example, in Monterrey businesses spearheaded, and funded the creation of a new state police force in association with universities and civil society (Conger 2014). Thus, we find little reason to believe that businesses in Mexico would systematically refuse to provide truthful answers regarding their extortion victimization experiences to the ENVE. As such, we believe that the extortion estimates captured by the ENVE suffer no more than the well-known measurement limitations shared by all well conducted victimization surveys ( see Lynch 2006, p. 245; Skogan 1986; UNODC/UNECE 2010). 

Lastly, other important limitations refer to the temporal horizon imposed by the cross-sectional design based on a 1 year period. First, by collapsing the temporal scale to 1 year, the study cannot capture the time-course of repeat victimization (Johnson et al. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1145 

1997)—i.e. it cannot measure how extortion risks change immediately after a victimization incident. Second, the reference period artificially imposes a time-window on repeat victimization—i.e. some extortions early in the period may be repeats of extortions that took place before the period began and some extortions at the end of the period may have repeats after it ends—which would lead to an undercounting of repeat victimization (Farrell and Pease 1993 , p. 19). Such limitations are difficult to overcome with the current data, though perhaps future iterations of the ENVE could incorporate an “embedded panel” (Hopkins and Tilley 2001) to address some temporal variation. 

In conclusion, this study applied a novel modeling strategy—the multilevel negative binomial-logit hurdle model—to identify whether the processes that lead to extortion prevalence are the same as those that lead to extortion concentration, as tends to be considered in the repeat victimization literature. The findings support the use of the hurdle model over the negative binomial model. Thus, studies on crimes where repeats are thought to be strongly influenced by event dependence mechanisms (such as domestic violence), would do well to test whether the hurdle model is a better fit. Furthermore, the study expands the crime concentration literature by focusing on a non-traditional crime type (extortion) in a new context (Mexico). Findings highlight areas that need to be researched further. These include, for example, analyzing area-level effects at smaller spatial resolutions, and directly discriminating between risk heterogeneity and event dependence as sources of risk for repeat extortion using longitudinal data. In addition, a follow up qualitative study of the most chronically victimized studies would further enhance our understanding of repeat extortion victimization. Hopefully, subsequent studies will find in this paper a useful starting point to build upon and develop finer insights. 

**Acknowledgements** PRES received funding from Mexico’s National Council of Science and Technology (CONACYT) and the State of Nuevo Léon (382181) and Mexico’s Secretary of Education (BC-7698, BC-6225, BC-4629, BC-2974). The authors thank Mexico’s National Statistics Agency (INEGI) for providing access to the data used in this study. We also thank the anonymous reviewers for their helpful comments and suggestions. 

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/. 

## **Appendix** 

See Tables 6, 7 and 8. 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1146 

**Table 6** Model estimates (log scale) and robustness checks for multilevel negative binomial (MNB) models 

||_MNB_|||
|---|---|---|---|
||Original (SE)|Bootstrap (SE)|Excluding outliers (SE)|
|Intercept|− 2.28*** (0.12)|− 2.28*** (0.14)|− 2.31*** (0.12)|
|_Business-level variables_||||
|Corruption victimizations|0.28*** (0.04)|0.28*** (0.07)|0.28*** (0.04)|
|_Business age (base: 0–5)_||||
|6–9|0.29*** (0.08)|0.29* (0.12)|0.30*** (0.08)|
|10–14|0.40*** (0.08)|0.40*** (0.11)|0.35*** (0.08)|
|15–23|0.43*** (0.08)|0.43*** (0.11)|0.44*** (0.08)|
|24–212|0.44*** (0.08)|0.44*** (0.10)|0.41*** (0.08)|
|_Business type (base: retail)_||||
|Mining|− 0.33 (0.44)|− 0.33 (0.32)|− 0.31 (0.44)|
|Construction|− 0.02 (0.14)|0.02 (0.14)|0.02 (0.14)|
|Manufacturing|− 0.21* (0.08)|− 0.21** (0.08)|− 0.18* (0.08)|
|Wholesale|0.06 (0.10)|0.06 (0.15)|− 0.00 (0.10)|
|Transport|− 0.03 (0.15)|− 0.03 (0.14)|0.01 (0.15)|
|Media|− 1.54*** (0.37)|− 1.54*** (0.31)|− 1.49*** (0.37)|
|Finance|0.20 (0.24)|0.20 (0.27)|0.21 (0.23)|
|Real estate|0.18 (0.21)|0.18 (0.15)|0.18 (0.20)|
|Prof. services|0.23 (0.15)|0.23 (0.20)|0.23 (0.15)|
|Maintenance|− 0.32* (0.15)|− 0.32 (0.20)|− 0.29 (0.15)|
|Education|− 0.15 (0.14)|− 0.15 (0.18)|− 0.12 (0.14)|
|Health|0.14 (0.13)|0.14 (0.12)|0.15 (0.13)|
|Leisure|− 0.20 (0.25)|− 0.20 (0.24)|− 0.20 (0.25)|
|Hotels Rest Bar|0.39*** (0.08)|0.39*** (0.09)|0.36*** (0.08)|
|Other|− 0.15 (0.10)|− 0.15 (0.12)|− 0.15 (0.09)|
|_Business size (base = large)_||||
|Medium|0.11 (0.09)|0.11 (0.12)|0.07 (0.09)|
|Small|0.02 (0.09)|0.02 (0.10)|0.07 (0.09)|
|Micro|− 0.69*** (0.08)|− 0.69*** (0.10)|− 0.64*** (0.08)|
|_State-level variables_||||
|Corruption prevalence (log)|0.36* (0.17)|0.36*** (0.09)|0.40* (0.17)|
|Weapon crimes (log)|0.45*** (0.11)|0.45*** (0.06)|0.44*** (0.11)|
|Drug crimes (log)|− 0.31*** (0.08)|− 0.31*** (0.04)|− 0.31*** (0.08)|
|N businesses (log)|− 0.45 (0.29)|− 0.45** (0.15)|− 0.49 (0.29)|
|Population (log)|− 0.10 (0.11)|− 0.10 (0.05)|− 0.11 (0.11)|
|Competitiveness index|− 0.02** (0.01)|− 0.02*** (0.00)|− 0.02** (0.01)|
|‘Rule of law’ index|− 0.00 (0.01)|− 0.00 (0.00)|− 0.00 (0.01)|
|n|28161|28161|28158|
|Groups|32|32|32|



> ∗∗∗ _p <_ 0.001 , ∗∗ _p <_ 0.01 , ∗ _p <_ 0.05 . Bootstrap repetitions: 99 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1147 

**Table 7** Model estimates (log scale) and robustness checks for multilevel logit (ML) models 

||_Hurdle: ML_|||
|---|---|---|---|
||Original (SE)|Bootstrap (SE)|Excluding outliers (SE)|
|Intercept|− 2.84*** (0.12)|− 2.84*** (0.09)|− 2.84*** (0.12)|
|_Business-level variables_||||
|Corruption victimizations|0.11*** (0.02)|0.11** (0.04)|0.11*** (0.02)|
|_Business age (base: 0–5)_||||
|6–9|0.33*** (0.07)|0.33*** (0.08)|0.33*** (0.07)|
|10–14|0.36*** (0.08)|0.36*** (0.07)|0.35*** (0.08)|
|15–23|0.45*** (0.07)|0.45*** (0.07)|0.45*** (0.07)|
|24–212|0.35*** (0.08)|0.35*** (0.07)|0.35*** (0.08)|
|_Business type (base: retail)_||||
|Mining|− 0.14 (0.40)|− 0.14 (0.41)|− 0.14 (0.40)|
|Construction|0.08 (0.12)|0.08 (0.12)|0.08 (0.12)|
|Manufacturing|− 0.17* (0.08)|− 0.17** (0.07)|− 0.17* (0.08)|
|Wholesale|0.07 (0.09)|0.07 (0.10)|0.07 (0.09)|
|Transport|0.23 (0.12)|0.23 (0.13)|0.23 (0.12)|
|Media|− 1.19** (0.36)|− 1.19** (0.38)|− 1.19** (0.36)|
|Finance|0.15 (0.21)|0.15 (0.19)|0.15 (0.21)|
|Real estate|0.26 (0.18)|0.26 (0.17)|0.26 (0.18)|
|Prof. services|0.18 (0.13)|0.18 (0.11)|0.18 (0.13)|
|Maintenance|− 0.04 (0.14)|− 0.04 (0.12)|− 0.04 (0.14)|
|Education|− 0.14 (0.12)|− 0.14 (0.12)|− 0.14 (0.12)|
|Health|0.11 (0.11)|0.11 (0.12)|0.11 (0.11)|
|Leisure|− 0.11 (0.23)|− 0.11 (0.17)|− 0.11 (0.23)|
|Hotels rest bar|0.36*** (0.07)|0.36*** (0.07)|0.36*** (0.07)|
|Other|− 0.15 (0.09)|− 0.15 (0.11)|− 0.15 (0.09)|
|_Business size (base = large)_||||
|Medium|0.21* (0.09)|0.21* (0.09)|0.21* (0.09)|
|Small|0.35*** (0.08)|0.35*** (0.08)|0.35*** (0.08)|
|Micro|− 0.35*** (0.08)|− 0.35*** (0.09)|− 0.35*** (0.08)|
|_State-level variables_||||
|Corruption prevalence (log)|0.48** (0.18)|0.48*** (0.08)|0.48** (0.18)|
|Weapon crimes (log)|0.44*** (0.12)|0.44*** (0.05)|0.44*** (0.12)|
|Drug crimes (log)|− 0.26** (0.09)|− 0.26*** (0.04)|− 0.26** (0.09)|
|<br>N businesses (log)|− 0.53 (0.31)|− 0.53*** (0.14)|− 0.53 (0.31)|
|Population (log)|− 0.15 (0.12)|− 0.15** (0.06)|− 0.15 (0.12)|
|Competitiveness index|− 0.02* (0.01)|− 0.02*** (0.00)|− 0.02* (0.01)|
|‘Rule of law’ index|− 0.00 (0.01)|− 0.00 (0.00)|− 0.00 (0.01)|
|n|28161|28161|28158|
|Groups|32|32|32|



> ∗∗∗ _p <_ 0.001 , ∗∗ _p <_ 0.01 , ∗ _p <_ 0.05 . Bootstrap repetitions: 99 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1148 

**Table 8** Model estimates (log scale) and robustness checks for multilevel truncated negative binomial (MTNB) models 

||_Hurdle: MTNB_|||
|---|---|---|---|
||Original (SE)|Bootstrap (SE)|Excluding outliers (SE)|
|Intercept|− 4.37*** (0.22)|− 4.37*** (0.31)|− 4.45*** (0.22)|
|_Business-level variables_||||
|Corruption victimizations|0.14*** (0.04)|0.14* (0.07)|0.15*** (0.04)|
|Business age (base: 0–5)||||
|6–9|0.09 (0.16)|0.09 (0.21)|0.08 (0.16)|
|10–14|0.14 (0.17)|0.14 (0.20)|0.01 (0.17)|
|15–23|0.18 (0.16)|0.18 (0.20)|0.16 (0.16)|
|24–212|0.31 (0.16)|0.31 (0.24)|0.22 (0.16)|
|Business type (base: retail)||||
|Mining|− 0.29 (0.85)|− 0.29 (0.87)|− 0.31 (0.85)|
|Construction|− 0.41 (0.26)|− 0.41 (0.29)|− 0.29 (0.26)|
|Manufacturing|− 0.14 (0.16)|− 0.14 (0.21)|− 0.08 (0.16)|
|Wholesale|0.05 (0.18)|− 0.05 (0.32)|− 0.09 (0.18)|
|Transport|− 0.71** (0.27)|− 0.71* (0.31)|− 0.58* (0.27)|
|Media|− 1.41 (0.85)|− 1.41 (0.72)|− 1.31 (0.85)|
|Finance|− 0.25 (0.47)|− 0.25 (0.49)|− 0.21 (0.46)|
|Real estate|− 0.03 (0.39)|− 0.03 (0.40)|− 0.04 (0.39)|
|Prof. services|− 0.14 (0.29)|− 0.14 (0.28)|− 0.13 (0.29)|
|Maintenance|− 0.93** (0.31)|− 0.93* (0.43)|− 0.83** (0.31)|
|Education|− 0.24 (0.26)|− 0.24 (0.38)|− 0.17 (0.26)|
|Health|0.05 (0.24)|0.05 (0.30)|0.06 (0.24)|
|Leisure|− 0.04 (0.51)|− 0.04 (0.71)|− 0.05 (0.50)|
|Hotels rest bar|0.30 (0.16)|0.30 (0.21)|0.26 (0.16)|
|Other|0.08 (0.20)|0.08 (0.31)|0.08 (0.20)|
|Business size (base = large)||||
|Medium|− 0.25 (0.18)|− 0.25 (0.24)|− 0.33 (0.18)|
|Small|− 0.72*** (0.16)|− 0.72*** (0.21)|− 0.61*** (0.16)|
|Micro|− 1.08*** (0.16)|− 1.08*** (0.19)|− 0.96*** (0.16)|
|_State-level variables_||||
|Corruption prevalence (log)|− 0.27 (0.29)|− 0.27 (0.19)|− 0.13 (0.28)|
|Weapon crimes (log)|0.14 (0.19)|0.14 (0.13)|0.14 (0.18)|
|Drug crimes (log)|− 0.28 (0.15)|− 0.28** (0.10)|− 0.29* (0.15)|
|N businesses (log)|0.31 (0.51)|0.31 (0.35)|0.13 (0.50)|
|Population (log)|0.15 (0.20)|0.15 (0.08)|0.14 (0.19)|
|Competitiveness index|− 0.02 (0.01)|− 0.02** (0.01)|− 0.02 (0.01)|
|‘Rule of law’ index|0.01 (0.01)|0.01 (0.01)|0.01 (0.01)|
|n|2266|2266|2263|
|Groups|32|32|32|



> ∗∗∗ _p <_ 0.001 , ∗∗ _p <_ 0.01 , ∗ _p <_ 0.05 . Bootstrap repetitions: 99 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1149 

## **References** 

Aguirre J, Herrera HA (2013) Institutional weakness and organized crime in Mexico: the case of Michoacán. Trends Organ Crime 16(2):221–238. https ://doi.org/10.1007/s1211 7-013-9197-1 

- Andresen MA, Curman ASN, Linning SJ (2016) The trajectories of crime at places: understanding the patterns of disaggregated crime types. J Quant Criminol 33(3):427–449. https ://doi.org/10.1007/s1094 0-016-9301-1 

- Andresen MA, Linning SJ, Malleson N (2017) Crime at places and spatial concentrations: exploring the spatial stability of property crime in Vancouver BC, 2003–2013. J Quant Criminol 33(2):255–275. https ://doi.org/10.1007/s1094 0-016-9295-8 

- Arnold TA, Emerson JW (2011) Nonparametric goodness-of-fit tests for discrete null distributions. R J 3(2):34–39. http://journ al.r-proje ct.org/archi ve/2011-2/RJour nal_2011-2_Arnol d+Emers on.pdf 

- Arulampalam W, Booth AL (1997) Who gets over the training hurdle? A study of the training experiences of young men and women in Britain. J Popul Econ 10(2):197–217. https ://doi.org/10.1007/ s0014 80050 038 

- Asmundo A, Lisciandra M (2008) The cost of protection racket in Sicily. Glob Crime 9(3):221–240. https :// doi.org/10.1080/17440 57080 22543 38 

- Astorga L (2012) México: Organized crime politics and insecurity. In: Segel D, van de Bunt H (eds) Traditional organized crime in the modern world. Springer, New York, NY, pp 149–166. https ://doi. org/10.1007/978-1-4614-3212-8_8 

- Baller RD, Zevenbergen MP, Messner SF (2009) The heritage of herding and southern homicide: examining the ecological foundations of the code of honor thesis. J Res Crime Delinq 46(3):275–300. https ://doi. org/10.1177/00224 27809 33516 4 

- Bell BA, Ferron JM, Kromrey JD (2008) Cluster size in multilevel models: the impact of sparse data structures on point and interval estimates in two-level models. JSM SRMS, pp 1122–1129 

- Berk R, MacDonald JM (2008) Overdispersion and poisson regression. J Quant Criminol 24(3):269–284. https ://doi.org/10.1007/s1094 0-008-9048-4 

- Bernasco W (2008) Them again?: Same-offender involvement in repeat and near repeat burglaries. Eur J Criminol 5(4):411–431. https ://doi.org/10.1177/14773 70808 09512 4 

- Best J (1982) Crime as strategic interaction: the social organization of extortion. J Contemp Ethnogr 11(1):107–128. https ://doi.org/10.1177/08912 41682 01100 105 

- Biderman AD (1980) Notes on measurement by crime victimization surveys. In: Fienberg SE, Reiss AJ (eds) Indicators of crime and criminal justice: quantitative studies. US Department of Justice, Bureau of Justice Statistics, Washington DC, pp 29–32 

- Bolker B, Skaug H, Magnusson A, Nielsen A (2012) Getting started with the glmmADMB package. http:// glmma dmb.r-forge .r-proje ct.org/glmmA DMB.pdf 

- Bouloukos AC, Farrell G (1997) On the displacement of repeat victimization. In: Newman G, Clarke RV, Shoham SG (eds) Rational choice and situational crime prevention. Dartmouth Press, Aldershot 

- Bowers KJ, Hirschfield A, Johnson SD (1998) Victimization revisited: a case study of non-residential repeat burglary on merseyside. Br J Criminol 38(3):429–452. http://bjc.oxfor djour nals.org/conte nt/38/3/429. abstr act 

- Brantingham P, Brantingham P (1993) Environment, routine and situation: toward a pattern theory of crime. In: Clarke RV, Felson M (eds) Routine activity and rational choice. Transaction publishers, New Brunswick, pp 259–294 

- Brantingham P, Brantingham P (2011) Crime pattern theory. In: Wortley R, Mazerolle L (eds) Environmental criminology and crime analysis. Routledge, Milton Park, UK, pp 78–94 

- Brantingham P, Brantingham P, Taylor W (2005) Situational crime prevention as a key component in embedded crime prevention. Can J Criminol Crim 47(2):271–292 

- Broadhurst RG, Bouhours B, Bacon-Shone J, Bouhours T (2011) Business and the risk of crime in China, vol (3). ANU E Press, Canberra, Australia 

- Brophy S (2008) Mexico: Cartels, corruption and cocaine: a profile of the gulf cartel. Glob Crime 9(3):248– 261. https ://doi.org/10.1080/17440 57080 22543 53 

- Brown CJ, Pagán JA, Rodríguez-Oreggia E (2005) The decision-making process of health care utilization in Mexico. Health Policy 72(1):81–91. https ://doi.org/10.1016/j.healt hpol.2004.06.008 

- Bunker RJ (2013) Introduction. The Mexican cartels. Organized crime vs. criminal insurgency. Trends Organ Crime 16(2):129–137. https ://doi.org/10.1007/s1211 7-013-9194-4 

- Burrows J, Hopkins M (2005) Business and crime. In: Tilley N (ed) Handbook of crime prevention and community safety. Willan, Cullompton, Devon, pp 486–515 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1150 

Calderon G, Robles G, Diaz-Cayeros A, Magaloni B (2015) The beheading of criminal organizations and the dynamics of violence in Mexico. J Confl Resol 59(8):1455–1485. https ://doi.org/10.1177/00220 02715 58705 3 

Cameron AC, Trivedi PK (2013) Regression analysis of count data. Cambridge University Press, Cambridge, UK 

- Camp RA (2012) The democratic transformation of Mexican politics. In: Camp RA (ed) The Oxford handbook of Mexican politics. Oxford University Press, Oxford UK. https ://doi.org/10.1093/oxfor dhb/97801 95377 385.013.0001 

Campbell L (2013) Organised crime and the law: a comparative analysis. Hart Publishing, Oxford 

- Cantoni E, Mills Flemming J, Welsh AH (2017) A random-effects hurdle model for predicting bycatch of endangered marine species. Ann Appl Stat 11(4):2178–2199. https ://doi.org/10.1214/17-AOAS1 074 

- Cantor D, Lynch JP (2000) Self-report surveys as measures of crime and criminal victimization. In: Measurement and analysis of crime and justice, Vol 4, National Institute of Justice NCJ-182411, pp 85–138 

Chin K (2000) Chinatown gangs: extortion, enterprise, and ethnicity. Oxford University Press, Oxford, UK 

- Chin K, Fagan J, Kelly RJ (1992) Patterns of Chinese gang extortion. Justice Q 9(4):625–646. https ://doi. org/10.1080/07418 82920 00915 81 

- Clarke RV, Cornish DB (2017) The rational choice perspective. In: Wortley R, Townsley M (eds) Environmental criminology and crime analysis. Routledge, Milton Park, UK, pp 29–61 

- Cohen LE, Felson M (1979) Social change and crime rate trends: a routine activity approach. Am Sociol Rev 44(4):588–608. 

- Colman AM (2015) difference threshold. In: A dictionary of psychology, 4th edn. Oxford University Press. https ://www.oxfor drefe rence .com/view/10.1093/acref /97801 99657 681.001.0001/acref -97801 99657 681-e-2291 

- Confcommercio-GFK Eurisko (2007) La mappa della criminalità regione per regione. http://www.confc ommer cio.it/-/-la-mappa -della -crimi nalit a-regio ne-per-regio ne- 

- Conger L (2014) The private sector and public security: the cases of Ciudad Juárez and monterrey. In: Shirk DA, Wood D, Olson EL (eds) Building resilient communities in Mexico: civic responses to crime and violence. Wilson Center Mexico Institute and University of San Diego Justice in Mexico Project, Washington DC, pp 173–209 

- Corcoran P (2012) Zetas bruised a year after Casino Royale, but monterrey still suffers. InSight Crime: investigation and analysis of organized crime. http://www.insig htcri me.org/news-analy sis/zetas -bruis ed-year-after -casin o-royal e-monte rrey-suffe rs 

- Corcoran P (2013) Mexico’s shifting criminal landscape: changes in gang operation and structure during the past century. Trends Organ Crime 16(3):306–328. https ://doi.org/10.1007/s1211 7-013-9190-8 

- Cornish DB, Clarke RV (1985) Modeling offender’s decisions: a framework for research and policy. Crime Justice 6:147–185 

- Cornish DB, Clarke RV (1987) Understanding crime displacement: an application of rational choice theory. Criminology 25(4):933–948. https ://doi.org/10.1111/j.1745-9125.1987.tb008 26.x 

- Correa-Cabrera G, Keck M, Nava J (2015) Losing the monopoly of violence: the state, a drug war and the paramilitarization of organized crime in Mexico (2007–2010). State Crime J 4(1):77–95 

- Curman ASN, Andresen MA, Brantingham PJ (2015) Crime and place: a longitudinal examination of street segment patterns in Vancouver. BC. J Quant Criminol 31(1):127–147. https ://doi. org/10.1007/s1094 0-014-9228-3 

- Daigle LE, Fisher BS, Cullen FT (2008) The violent and sexual victimization of college women: is repeat victimization a problem? J Interpers Violence 23(9):1296–1313. https ://doi.org/10.1177/08862 60508 31429 3 

- Dell M (2014) Trafficking networks and the Mexican drug war. Unpublished working paper http://schol 

   - ar.harva rd.edu/dell/publi catio ns/traffi ckin g-netwo rks-and-mexic an-drug-war-0 

- Di Gennaro G, La Spina A (2016) The costs of illegality: a research programme. Glob Crime 17(1):1– 20. https ://doi.org/10.1080/17440 572.2015.11286 21 

- Dickenson M (2014) The impact of leadership removal on Mexican drug trafficking organizations. J Quant Criminol 30(4):651–676. https ://doi.org/10.1007/s1094 0-014-9218-5 

- Dugato M (2014) Analyzing bank robbery in Italy. In: Caneppele S, Calderoni F (eds) Organized crime, corruption and crime prevention. Springer, London, pp 115–125. https ://doi.org/10.1007/978-3319-01839 -3_14 

- Duran-Martinez A (2015) To kill and tell? State power, criminal competition, and drug violence. J Conf Resol 59(8):1377–1402. https ://doi.org/10.1177/00220 02715 58704 7 

- Díaz-Cayeros A, Magaloni B, Romero V (2015) Caught in the crossfire: the geography of extortion and police corruption in Mexico. In: Rose-Ackerman S, Lagunes P (eds) Greed, Corruption, and the modern state: essays in political economy. Edward Elgar Publishing., Cheltenham UK 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1151 

Elsenbroich C, Badham J (2016) The extortion relationship: a computational analysis. J Artif Soc Soc Simul. https ://doi.org/10.18564 /jasss .3223 

Everson S, Pease K (2001) Crime against the same person and place: detection opportunity and offender targeting. In: Farrell G, Pease K (eds) Crime prevention studies, vol 12: repeat victimization. Criminal Justice Press, Monsey, NY, pp 199–220. http://www.popce nter.org/libra ry/crime preve ntion /volum e_12/11-Evers on.pdf 

extortion (2010) Oxford dictionary of English. Oxford University Press, Oxford, UK. http://www.oxfor drefe rence .com/view/10.1093/acref /97801 99571 123.001.0001/m_en_gb028 2380 

- extortion (2017) Encyclopædia Britannica. https ://www.brita nnica .com/topic /extor tion 

- Farrell G, Bouloukos AC (2001) International overview: a cross-national comparison of rates of repeat victimization. In: Farrell G, Bouloukos AC (eds) Crime Prevention studies, vol 12. Repeat victimization. Criminal Justice Press, Moonsey, NY, pp 5–26 

- Farrell G, Pease K (1993) Once bitten, twice bitten: Repeat victimisation and its implications for crime prevention. Home Office, Police Research Group, London 

- Farrell G, Pease K (2007) The sting in the tail of the british crime survey: multiple victimisations. In: Hough M, Maxfield M (eds) Crime prevention studies, vol 22. Surveying crime in the 21st century. Criminal Justice Press, Moonsey, NY, pp 3–53 

- Farrell G, Pease K (2011) Repeat victimisation. In: Wortley R, Mazerolle L (eds) Environmental criminology and crime analysis. Routledge, Milton Park, UK, pp 117–135 

- Farrell G, Pease K (2014) Repeat victimization. In: Bruinsma G, Weisburd D (eds) Encyclopedia of criminology and criminal justice. Springer, New York, pp 4371–4381. https ://doi. org/10.1007/978-1-4614-5690-2_128 

- Farrell G, Phillips C, Pease K (1995) Like taking candy-why does repeat victimization occur. Brit J Criminol 35:384 

- Farrell G, Tseloni A, Pease K (2005) Repeat victimization in the ICVS and the NCVS. Crime Prev Comm Saf 7(3):7–18 

- Felson M (2017) Routine activity approach. In: Wortley R, Townsley M (eds) Environmental criminology and crime analysis. Routledge, Milton Park, UK, pp 87–97 

- Fournier DA, Skaug HJ, Ancheta J, Ianelli J, Magnusson A, Maunder MN, Nielsen A, Sibert J (2012) AD model builder: using automatic differentiation for statistical inference of highly parameterized complex nonlinear models. Optim Methods Softw 27(2):233–249. https ://doi.org/10.1080/10556 788.2011.59785 4 

- Fox J, Monette G (1992) Generalized collinearity diagnostics. J Am Stat Assoc 87(417):178–183. https :// doi.org/10.2307/22904 67 

- Frazzica G, La Spina A, Scaglione A (2013) Mafia-type organizations in Italy: diffusion, impact on the private sector and research paths. In: Mugellini G (ed) Measuring and analyzing crime against the private sector: international experiences and the Mexican practice. INEGI, México DF, pp 97–128 

- Frye T, Zhuravskaya E (2000) Rackets, regulation, and the rule of law. J Law Econ Organ 16(2):478–502. https ://doi.org/10.1093/jleo/16.2.478 

- Gambetta D (1993) The Sicilian Mafia: the business of private protection. Harvard University Press, Cambridge, MA 

- Gill M (1998) The victimisation of business: indicators of risk and the direction of future research. Int Rev Vict 6(1):17–28. https ://doi.org/10.1177/02697 58098 00600 102 

- Goldstein H (2011) Multilevel statistical models. Willey, Chichester, UK. https ://doi.org/10.1002/97804 70973 394 

- Gottfredson MR (1986) Substantive contributions of victimization surveys. Crime Justice 7:251–287. http:// www.jstor .org/stabl e/11475 19 

- Gray BR (2005) Selecting a distributional assumption for modelling relative densities of benthic macroinvertebrates. Ecol Model 185(1):1–12. https ://doi.org/10.1016/j.ecolm odel.2004.11.006 

- Grove L, Farrell G (2010) Repeat victimization. In: Fisher BS, Lab SP (eds) Encyclopedia of victimology and crime prevention, vol 2. Sage Publications, Thousand Oaks, pp 767–769. https ://doi. org/10.4135/97814 12979 993.n258 

- Guerrero-Gutiérrez E (2011) Security, drugs, and violence in Mexico: a survey. 7th North American Forum, Washington DC 

- Heckman JJ (1981) Statistical models for discrete panel data. In: Manski CF, McFadden D (eds) Structural analysis of discrete data with econometric applications. The MIT Press, Cambridge, MA, pp 114– 178. https ://eml.berke ley.edu/~mcfad den/discr ete.html 

- Hellemans S, Loeys T, Dewitte M, De Smet O, Buysse A (2015) Prevalence of intimate partner violence victimization and victims’ relational and sexual well-being. J Fam Violence 30(6):685–698. https :// doi.org/10.1007/s1089 6-015-9712-z 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1152 

Hepenstal S, Johnson SD (2010) The concentration of cash-in-transit robbery. Crime Prev Comm Saf 12(4):263–282 

- Hester R, Hartman TK (2017) Conditional race disparities in criminal sentencing: a test of the liberation hypothesis from a non-guidelines state. J Quant Criminol 33(1):77–100. https ://doi.org/10.1007/ s1094 0-016-9283-z 

- Hilbe JM (2011) Negative binomial regression. Cambridge University Press, Cambridge, UK Hilbe JM (2014) Modeling count data. Cambridge University Press, Cambridge, UK 

- Hindelang MJ, Gottfredson MR, Garofalo J (1978) Victims of personal crime: an empirical foundation for a theory of personal victimization. Ballinger, Cambridge, MA 

- Hope T, Norris PA (2012) Heterogeneity in the frequency distribution of crime victimization. J Quant Criminol 29(4):543–578. https ://doi.org/10.1007/s1094 0-012-9190-x 

- Hopkins M, Tilley N (2001) Once a victim, always a victim? A study of how victimisation patterns may change over time. Int Rev Vict 8(1):19–35. https ://doi.org/10.1177/02697 58001 00800 102 

- Hough M (1987) Offenders’ choice of target: Findings from victim surveys. J Quant Criminol 3(4):355–369. https ://doi.org/10.1007/BF010 66836 

- IMCO (2016) Índice de Competitividad Estatal 2016: Un puente entre dos Méxicos. Instituto Mexicano para la Competitividad, AC, Ciudad de Mexico 

- INEGI (2007) Sistema de Clasificación Industrial de América del Norte, México: SCIAN 2007. Instituto Nacional de Estadística y Geografía, México DF 

- INEGI (2014a) Boletín de Prensa 550/14: Encuesta Nacional de Victimización de Empresas 2014. Instituto Nacional de Estadística y Geografía, Aguascalientes, Ags., http://www.inegi .org.mx 

- INEGI (2014b) Encuesta Nacional de Victimización de Empresas 2014 ENVE: Documento metodológico sobre diseño muestral. Instituto Nacional de Estadística y Geografía, México DF 

- INEGI (2014c) Encuesta Nacional de Victimización de Empresas 2014 ENVE: Marco conceptual. Instituto Nacional de Estadística y Geografía, México DF 

- INEGI (2014d) Encuesta Nacional de Victimización de Empresas (ENVE - 2014): Cuestionario Principal. Instituto Nacional de Estadística y Geografía, México DF, http://www.inegi .org.mx 

- Jaimes Bello O, Vielma Orozco E (2013) Measuring crime against the private sector in Mexico: the crime against business national survey 2012 (ENVE). In: Mugellini G (ed) Measuring and analyzing crime against the private sector: international experiences and the Mexican practice. INEGI, México, DF 

- Johnson SD (2008) Repeat burglary victimisation: a tale of two theories. J Exp Criminol 4(3):215–240. https ://doi.org/10.1007/s1129 2-008-9055-3 

- Johnson SD (2014) How do offenders choose where to offend? Perspectives from animal foraging. Legal Criminol Psychol 19(2):193–210. https ://doi.org/10.1111/lcrp.12061 

- Johnson SD, Bowers KJ (2004) The burglary as clue to the future: the beginnings of prospective hotspotting. Eur J Criminol 1(2):237–255. https ://doi.org/10.1177/14773 70804 04125 2 

- Johnson SD, Bowers KJ (2010) Permeability and burglary risk: are Cul-de-Sacs safer? J Quant Criminol 26(1):89–111. https ://doi.org/10.1007/s1094 0-009-9084-8 

- Johnson SD, Bowers KJ, Hirschfield A (1997) New insights into the spatial and temporal distribution of repeat victimization. Brit J Criminol 37(2):224–241. http://bjc.oxfor djour nals.org/conte nt/37/2/224.abstr act 

- Johnson SD, Summers L, Pease K (2009) Offender as forager? A direct test of the boost account of victimization. J Quant Criminol 25(2):181–200. https ://doi.org/10.1007/s1094 0-008-9060-8 

- Jones NP (2013) The unintended consequences of kingpin strategies: kidnap rates and the Arellano-Félix organization. Trends Organ Crime 16(2):156–176. https ://doi.org/10.1007/s1211 7-012-9185-x 

- Jones NP (2016) Mexico’s illicit drug networks and the state reaction. Georgetown University Press, Washington, DC 

- Kelly RJ, Chin K, Fagan J (2000) Lucky money for little brother: the prevalence and seriousness of Chinese gang extortion. Int J Comp Appl Crim Justice 24(1):61–90. https ://doi.org/10.1080/01924 036.2000.96786 53 

- Kleemans ER (2001) Results of empirical research in the Netherlands. In: Farrell G, Pease K (eds) Crime prevention studies, vol 12. Repeat victimization. Criminal Justice Press, Monsey, NY, pp 53–68 

- Kleemans ER (2014) Theoretical perspectives on organized crime. In: Paoli L (ed) The Oxford handbook of organized crime. Oxford University Press, Oxford, UK, pp 32–52. https ://doi.org/10.1093/ oxfor dhb/97801 99730 445.013.005 

- Kleemans ER (2015) Criminal organization and transnational crime. In: Bruinsma G (ed) Histories of transnational crime. Springer, UK, pp 171–185. https ://doi.org/10.1007/978-1-4939-2471-4_8 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1153 

Kleemans ER (2018) Organized crime and places. In: Bruinsma GJN, Johnson SD (eds) Handbook of environmental criminology. Oxford University Press, Oxford, UK, pp 872–886. https ://doi. org/10.1093/oxfor dhb/97801 90279 707.013.24 

Konrad KA, Skaperdas S (1998) Extortion. Economica 65(260):461–477. https ://doi. org/10.1111/1468-0335.00141 

- Kuo SY, Cuvelier SJ, Sheu CJ, Zhao J (2012) The concentration of criminal victimization and patterns of routine activities. Int J Offender Ther Comp Criminol 56(4):573–598. https ://doi. org/10.1177/03066 24X11 40071 5 

- La Spina A (ed) (2008) I costi dell’illegalitá : Mafia ed estorsioni in Sicilia. Il Mulino, Bologna 

- La Spina A, Frazzica G, Punzo V, Scaglione A (2014) How mafia works. an analysis of the extortion racket system. In: Proceedings of ECPR general conference, Glasgow, UK. https ://ecpr.eu/Event s/ Paper Detai ls.aspx?Paper ID=22389 &Event ID=14 

- Lambert D (1992) Zero-inflated poisson regression, with an application to defects in manufacturing. Technometrics 34(1):1–14. https ://doi.org/10.2307/12695 47 

- Lauritsen JL (2010) Advances and challenges in empirical studies of victimization. J Quant Criminol 26(4):501–508. https ://doi.org/10.1007/s1094 0-010-9118-2 

- Lauritsen JL, Davis Quinet KF (1995) Repeat victimization among adolescents and young adults. J Quant Criminol 11(2):143–166 

- Lauritsen JL, Rezey ML (2018) Victimization trends and correlates: macro- and microinfluences and new directions for research. Annu Rev Criminol 1:103–121. https ://doi.org/10.1146/annur ev-crimi nol-03231 7-09220 2 

- Lee Y, Eck JE, SooHyun O, Martinez NN (2017) How concentrated is crime at places? A systematic review from 1970 to 2015. Crime Sci 6(1):6. https ://doi.org/10.1186/s4016 3-017-0069-x 

- Lynch JP (2006) Problems and promise of victimization surveys for cross-national research. Crime Justice 34(1):229–287. https ://doi.org/10.1086/50267 0 

- Lynch JP, Berbaum M, Planty M (1998) Investigating repeated victimization with the NCVS: final report. Doc. 193415. National Institute of Justice, Washington, DC 

- MacDonald JM, Lattimore PK (2010) Count models in criminology. In: Piquero AR, Weisburd D (eds) Handbook of quantitative criminology. Springer, New York, NY, pp 683–698. https ://doi. org/10.1007/978-0-387-77650 -7_32 

- Matthews R, Pease C, Pease K (2001) Repeated bank robbery: themes and variations. In: Farrell G, Pease K (eds) Crime prevention studies, vol 12. Repeat victimization. Criminal Justice Press, Moonsey, NY, pp 153–164 

- Maxfield MG (1987a) Household composition, routine activity, and victimization: a comparative analysis. J Quant Criminol 3(4, Special Issue: Lifestyle and Routine Activity Theories of Crime):301– 320. https ://doi.org/10.2307/23365 568 

- Maxfield MG (1987b) Lifestyle and routine activity theories of crime: empirical studies of victimization, delinquency, and offender decision-making. J Quant Criminol 3(4):275–282. https ://doi. 

   - org/10.1007/BF010 66831 

- Mayhew P, van Dijk J (2014) International crime victimization survey. In: Bruinsma GJN, Weisburd D (eds) Encyclopedia of criminology and criminal justice. Springer, New York, NY, pp 2602–2614. https ://doi.org/10.1007/978-1-4614-5690-2_444 

- McCulloch CE, Neuhaus JM (2011) Misspecifying the shape of a random effects distribution: why getting it wrong may not matter. Stat Sci 26(3):388–402. https ://doi.org/10.1214/11-STS36 1 

- McIntosh M (1973) The growth of racketeering. Econ Soc 2(1):35–69. https ://doi.org/10.1080/03085 14730 00000 02 

- Melo SN, Matias LF, Andresen MA (2015) Crime concentrations and similarities in spatial crime patterns in a Brazilian context. Appl Geogr 62:314–324. https ://doi.org/10.1016/j.apgeo g.2015.05.012 

- Miethe TD, McDowall D (1993) Contextual effects in models of criminal victimization. Soc Forces 71(3):741–759. https ://doi.org/10.2307/25798 93 

- Miethe TD, Meier RF (1990) Opportunity, choice, and criminal victimization: a test of a theoretical model. J Res Crime Delinq 27(3):243–266. https ://doi.org/10.1177/00224 27890 02700 3003 

- Miethe TD, Stafford MC, Long JS (1987) Social differentiation in criminal victimization: a test of routine activities/lifestyle theories. Am Sociol Rev 52(2):184–194. https ://doi.org/10.2307/20954 47 

- Min Y, Agresti A (2005) Random effect models for repeated measures of zero-inflated count data. Stat Model 5(1):1–19. https ://doi.org/10.1191/14710 82x05 st084 oa 

- Morgan F (2001) Repeat burglary in a Perth suburb: indicator of short- term or long-term risk? In: Farrell G, Pease K (eds) Crime prevention studies, vol 12. Repeat victimization. Criminal Justice Press, Monsey, NY, pp 83–118 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1154 

Morris SD (2013) Drug trafficking, corruption, and violence in Mexico: mapping the linkages. Trends Organ Crime 16(2):195–220. https ://doi.org/10.1007/s1211 7-013-9191-7 

Mugellini G (ed) (2012) Le imprese vittime di criminalità in Italia. Transcrime, Universitá Cattolica Del Sacro Cuore Di Milano E Universitá Degli Studi Di Trento, Trento Mullahy J (1986) Specification and testing of some modified count data models. J Econom 33(3):341– 365. https ://www.scien cedir ect.com/scien ce/artic le/pii/ 03044 07686 90002 3 

- Nelson JF (1980) Multiple victimization in American cities: a statistical analysis of rare events. Am J Sociol 85(4):870–891. https ://doi.org/10.2307/27787 10 

Osborn DR, Tseloni A (1998) The distribution of household property crimes. J Quant Criminol 14(3):307–330. https ://doi.org/10.1023/A:10230 86530 548 Osborn DR, Ellingworth D, Hope T, Trickett A (1996) Are repeatedly victimized households different? J Quant Criminol 12(2):223–245. https ://doi.org/10.1007/BF023 54416 

- Osorio J (2015) The contagion of drug violence: spatiotemporal dynamics of the Mexican war on drugs. J Conflict Resolut 59(8):1403–1432. https ://doi.org/10.1177/00220 02715 58704 8 

- O’brien RM (2007) A caution regarding rules of thumb for variance inflation factors. Qual Quant 41(5):673–690. https ://doi.org/10.1007/s1113 5-006-9018-6 

- Paoli L (2002) The paradoxes of organized crime. Crime Law Soc Change 37(1):51–97. https ://doi. org/10.1023/A:10133 55122 531 

- Paoli L (2014) Introduction. In: Paoli L (ed) The Oxford handbook of organized crime. Oxford University Press, Oxford UK. https ://doi.org/10.1093/oxfor dhb/97801 99730 445.013.034 

- Paoli L, Vander Beken T (2014) Organized crime: a contested concept. In: Paoli L (ed) The Oxford handbook of organized crime. Oxford University Press, Oxford UK. https ://doi.org/10.1093/oxfor dhb/97801 99730 445.013.019 

- Sm P (2015) A study of over-dispersed household victimizations in South Korea: zero-inflated negative binomial analysis of Korean national crime victimization survey. Asian J Criminol 10(1):63–78. https ://doi.org/10.1007/s1141 7-015-9206-1 

- Sm P, Fisher BS (2015) Understanding the effect of immunity on over-dispersed criminal victimizations: zero-inflated analysis of household victimizations in the NCVS. Crime Delinq 63(9):1116–1145. https ://doi.org/10.1177/00111 28715 60753 4 

- Paul C, Schaefer AG, Clarke CP (2011) The challenge of violent drug-trafficking organizations. Rand Corporation, Santa Monica, CA 

- Pease K (1998) Repeat victimisation: taking stock. Home Office Police Research Group, London Pease K, Tseloni A (2014) Using modeling to predict and prevent victimization. Springer, New York, NY. https ://doi.org/10.1007/978-3-319-03185 -9 

- Perreault S, Sauvé J, Burns M (2010) Multiple victimization in Canada, 2004. Canadian Centre for Justice Statistics, Statistics Canada, Ottawa 

- Pires SF, Guerette RT, Stubbert CH (2014) The crime triangle of kidnapping for ransom incidents in Colombia, South America: A ’Litmus’ test for situational crime prevention. Br J Criminol 54(5):784–808. https ://doi.org/10.1093/bjc/azu04 4 

- Pitcher AB, Johnson SD (2011) Exploring theories of victimization using a mathematical model of burglary. J Res Crime Delinq 48(1):83–109. https ://doi.org/10.1177/00224 27810 38413 9 

- Pohlmeier W, Ulrich V (1995) An econometric model of the two-part decisionmaking process in the demand for health care. J Hum Resour 30(2):339–361. https ://doi.org/10.2307/14612 3 

- Polvi N, Looman T, Humphries C, Pease K (1991) The time course of repeat burglary victimization. Br J Criminol 31(4):411–414. https ://doi.org/10.1093/oxfor djour nals.bjc.a0481 38 

- R Core Development Team (2015) R: a language and environment for statistical computing. R Foundation for Statistical Computing. http://www.R-proje ct.org 

- Rand MR, Saltzman LE (2003) The nature and extent of recurring intimate partner violence against women in the United States. J Comp Fam Stud 34:137–149 

- Redacción (2010) Juárez es la ciudad más violenta del mundo. El Universal http://archi vo.eluni versa l.com. mx/notas /65095 6.html 

- Rios V (2012) Why did Mexico become so violent? A self-reinforcing violent equilibrium caused by competition and enforcement. Trends Organ Crime 16(2):138–155. https ://doi.org/10.1007/s1211 7-012-9175-z 

- Rios V (2015) How government coordination controlled organized crime: the case of Mexico’s cocaine markets. J Conf Resol 59(8):1433–1454. https ://doi.org/10.1177/00220 02715 58705 2 

- Rose CE, Martin SW, Wannemuehler KA, Plikaytis BD (2006) On the use of zero-inflated and hurdle models for modeling vaccine adverse event count data. J Biopharm Stat 16(4):463–481. https ://doi. org/10.1080/10543 40060 07193 84 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1155 

- Rosser G, Davies T, Bowers KJ, Johnson SD, Cheng T (2016) Predictive crime mapping: arbitrary grids or street networks? J Quant Criminol. https ://doi.org/10.1007/s1094 0-016-9321-x 

- Rydberg J, Carkin DM (2016) Utilizing alternate models for analyzing count outcomes. Crime Delinq 63(1):61–76. https ://doi.org/10.1177/00111 28716 67884 8 

- Rydberg J, Cassidy M, Socia KM (2017) Punishing the wicked: examining the correlates of sentence severity for convicted sex offenders. J Quant Criminol. https ://doi.org/10.1007/s1094 0-017-9360-y 

- Sagovsky A, Johnson SD (2007) When does repeat burglary victimisation occur? Aust N Z J Criminol 40(1):1–26. https ://doi.org/10.1375/acri.40.1.1 

- Salmi V, Kivivuori J, Lehti M (2013) Public disorder and business crime victimization in the retail sector. Security J. https ://doi.org/10.1057/sj.2012.56 

- Sansó-Rubert Pascual D (2017) Measuring organised crime: complexities of the quantitative and factorial analysis. In: Legind Larsen H, Blanco JM, Pastor Pastor R, Yager RR (eds) Using open data to detect organized crime threats: factors driving future crime. Springer, Cham, pp 25–44 

- Savona EU (2012) Italian Mafia’s asymmetries. In: Siegel D, van de Bunt H (eds) Traditional organized crime in the modern world: responses to socioeconomic change. Springer, New York, NY, pp 3–26. https ://doi.org/10.1007/978-1-4614-3212-8_1 

- Savona EU, Zanella M (2010) Extortion and organized crime. In: Natarajan M (ed) International crime and justice. Cambridge University Press, Cambridge UK. https ://doi.org/10.1017/CBO97 80511 76211 6.041 

- Savona EU, Sarno F (2014) Racketeering. In: Bruinsma GJN, Weisburd D (eds) Encyclopedia of criminology and criminal justice. Springer, New York, pp 4264–4273. https ://doi. org/10.1007/978-1-4614-5690-2_633 

- Schelling TC (1971) What is the business of organized crime? Am ar40(4):643–652. http://www.jstor .org/stabl e/41209 902 

- SESNSP (2015) Incidencia Delictiva. Secretariado Ejecutivo http://secre taria doeje cutiv o.gob.mx/index .php 

- Shirk DA, Wallman J (2015) Understanding Mexico’s drug violence. J Conf Resol 59(8):1348–1376. https ://doi.org/10.1177/00220 02715 58704 9 

- Shirk DA, Wood D, Olson EL (eds) (2014) Building Resilient Communities in Mexico: civic responses to crime and violence. Wilson Center Mexico Institute and University of San Diego Justice in Mexico Project, Washington DC 

- Sidebottom A (2013) Understanding and preventing crime in Malawi: an opportunity perspective. Ph.D. thesis, UCL (University College London) 

- Sidebottom A (2012) Repeat burglary victimization in Malawi and the influence of housing type and area-level affluence. Secur J 25(3):265–281. https ://doi.org/10.1057/sj.2011.22 

- Skaperdas S (2001) The political economy of organized crime: providing protection when the state does not. Econ Gov 2(3):173–202 

- Skogan WG (1986) From crime policy to victim policy: reorienting the justice system. Palgrave Macmillan, London, pp 80–116 

- Smith PH (2012) Mexican democracy in comparative perspective. In: Camp RA (ed) The Oxford handbook of Mexican politics. Oxford University Press, Oxford UK. https ://doi.org/10.1093/oxfor dhb/97801 95377 385.013.0004 

- SooHyun O, Martinez NN, Lee Y, Eck JE (2017) How concentrated is crime among victims? A systematic review from 1977 to 2014. Crime Sci 6(1):9. https ://doi.org/10.1186/s4016 3-017-0071-3 

- Sparks RF (1981a) Multiple victimization: evidence, theory, and future research. J Crim Law Criminol. https ://doi.org/10.2307/11430 14 

- Sparks RF (1981b) Surveys of victimization—an optimistic assessment. Crime Justice 3:1–60 

- Sparks RF, Genn HG, Dodd DJ (1977) Surveying victims: a study of the measurement of criminal victimization, perceptions of crime, and attitudes to criminal justice. Wiley, London 

- Stubbert CH, Pires SF, Guerette RT (2015) Crime science and crime epidemics in developing countries: a reflection on kidnapping for ransom in Colombia, South America. Crime Sci. https ://doi. org/10.1186/s4016 3-015-0034-5 

- Sung HE (2004) State failure, economic failure, and predatory organized crime: a comparative analysis. J Res Crime Delinq 41(2):111–129. https ://doi.org/10.1177/00224 27803 25725 3 

- Tilley N, Hopkins M (2008) Organized crime and local businesses. Criminol Crim Justice 8(4):443–459. https ://doi.org/10.1177/17488 95808 09646 9 

- Titus RM, Gover AR (2001) Personal frauds: the victims and the scams. In: Farrell G, Pease K (eds) Crime prevention studies, vol 12: repeat victimization. Criminal Justice Press, Monsey, NY, pp 133–151. https ://popce nter.asu.edu/sites /defau lt/files /libra ry/crime preve ntion /volum e_12/08-Titus .pdf 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1156 

TRANSCRIME (2009) Study on extortion racketeering the need for an instrument to combat activities of organised crime: final report. European Commission Trickett A, Osborn DR, Seymour J, Pease K (1992) What is different about high crime areas? Br J Criminol 32(1):81–89. http://bjc.oxfor djour nals.org/conte nt/32/1/81.abstr act 

- Tseloni A (1995) The modelling of threat incidence: evidence from the British crime survey. In: Dobash RE, Dobash RP, Noaks L (eds) Gender and crime. University of Wales Press, Cardiff, pp 269–294. http://irep.ntu.ac.uk/id/eprin t/1486/ 

- Tseloni A (2006) Multilevel modelling of the number of property crimes: Household and area effects. J R Stat Soc A Stat 169(2):205–233. https ://doi.org/10.1111/j.1467-985X.2005.00388 .x 

- Tseloni A, Farrell G (2002) Burglary victimization across Europe: the roles of prior victimization, micro and macro-level routine activities. In: Nieuwbeerta P (ed) Crime victimization in comparative perspective: results from the international crime victims survey, 1989–2000. Boom Juridische uitgevers, Den Haag, pp 141–161 

- Tseloni A, Pease K (2003) Repeat personal victimization ’Boosts’ or ’Flags’? Br J Criminol 43(1):196– 212. https ://doi.org/10.1093/bjc/43.1.196 

- Tseloni A, Pease K (2004) Repeat personal victimization: random effects, event dependence and unexplained heterogeneity. Brit J Criminol 44(6):931–945. https ://doi.org/10.1093/bjc/azh04 7 

- Tseloni A, Pease K (2005) Population inequality: the case of repeat crime victimization. Int Rev Vict 12(1):75–90. https ://doi.org/10.1177/02697 58005 01200 105 

- Tseloni A, Pease K (2014) Area and individual differences in personal crime victimization incidence: the role of individual, lifestyle/routine activities and contextual predictors. Int Rev Vict 21(1):3–29. https ://doi.org/10.1177/02697 58014 54799 1 

- Tseloni A, Osborn DR, Trickett A, Pease K (2002) Modelling property crime using the British crime survey. What have we learnt? Br J Criminol 42(1):109–128. https ://doi.org/10.1093/bjc/42.1.109 

- Tseloni A, Wittebrood K, Farrell G, Pease K (2004) Burglary victimization in England and Wales, the United States and the Netherlands: a cross-national comparative test of routine activities and lifestyle theories. Br J Criminol 44(1):66–91. https ://doi.org/10.1093/bjc/44.1.66 

- Tulyakov VA (2001) The dualism of business victimization and organized crime. Trends Organ Crime 6(3– 4):94–99. https ://doi.org/10.1007/s1211 7-001-1008-4 

- UNODC/UNECE (2010) Manual on victimization surveys. United Nations Office on Drugs and Crime and the United Nations Economic Commission for Europe, Geneva 

- Upton G, Cook I (2014a) Kolmogorov–Smirnov test. In: A dictionary of statistics. Oxford University Press, Oxford, UK. http://www.oxfor drefe rence .com/view/10.1093/acref / 97801 99679 188.001.0001/acref -97801 99679 188-e-868 

- Upton G, Cook I (2014b) Lorenz curve. In: A dictionary of statistics. Oxford University Press, Oxford, UK. http://www.oxfor drefe rence .com/view/10.1093/acref / 97801 99679 188.001.0001/acref -97801 99679 188-e-958 

- van Dijk J, Terlouw GJ (1996) An international perspective of the business community as victims of fraud and crime. Secur J 7(3):157–167 

- Varese F (2001) The Russian Mafia: private protection in a new market economy. Oxford University Press, Oxford. https ://doi.org/10.1093/01982 9736X .001.0001 

- Varese F (2011) Mafias on the move. Princeton University Press, Princeton, NJ. http://www.jstor .org/stabl e/j.ctt7t 96v 

- Varese F (2014) Protection and extortion. In: Paoli L (ed) The Oxford handbook of organized crime. Oxford University Press, Oxford, UK, pp 343–358. https ://doi.org/10.1093/oxfor dhb/97801 99730 445.013.020 

- Volkov V (2002) Violent entrepreneurs: the use of force in the making of Russian capitalism. Cornell University Press, Ithaca 

- von Lampe K (2005) Making the second step before the first: assessing organized crime. Crime Law Soc Change 42(4–5):227–259. https ://doi.org/10.1007/s1061 1-005-5305-8 

- von Lampe K (2016) Organized crime: analyzing illegal activities, criminal structures, and extra-legal governance. Sage Publications, Thousand Oaks 

- Wallace D, Eason JM, Lindsey AM (2015) The influence of incarceration and Re-entry on the availability of health care organizations in Arkansas. Health Justice 3(1):146. https ://doi.org/10.1186/s4035 2-015-0016-4 

- Wartell J, Gallagher K (2012) Translating environmental criminology theory into crime analysis practice. Policing 6(4):377–387. https ://doi.org/10.1093/polic e/pas02 0 

- Weisburd D (2015) The law of crime concentration and the criminology of place. Criminology 53(2):133– 157. https ://doi.org/10.1111/1745-9125.12070 

1 3 

Journal of Quantitative Criminology (2021) 37:1115–1157 

1157 

Weisburd D, Britt C (2014) Statistics in criminal justice. Springer, New York, NY. https ://doi. org/10.1007/978-1-4614-9170-5 

Wetzels P, Ohlemacher T, Pfeiffer C, Strobl R (1994) Victimization surveys: recent developments and perspectives. Eur J Crim Pol Res 2(4):14–35. https ://doi.org/10.1007/BF022 49437 Wilkinson T (2011) Suspect says Mexico casino fire set over unpaid extortion money. Los Angeles Times http://artic les.latim es.com/2011/aug/29/world /la-fg-mexic o-casin o-arres ts-20110 830 Wittebrood K, Nieuwbeerta P (2000) Criminal victimization during one’s life course: the effects of previous victimization and patterns of routine activities. J Res Crime Delinq 37(1):91–122. https ://doi. org/10.1177/00224 27800 03700 1004 

Young BJ, Furman W (2007) Interpersonal factors in the risk for sexual victimization and its recurrence during adolescence. J Youth Adolesc 37(3):297–309. https ://doi.org/10.1007/s1096 4-007-9240-0 Yu SSV, Maxfield MG (2014) Ordinary business: impacts on commercial and residential burglary. Brit J Criminol 54(2):298–320. https ://doi.org/10.1093/bjc/azt06 4 

**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

1 3 


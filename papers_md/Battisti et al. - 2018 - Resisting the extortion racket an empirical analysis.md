European Journal of Law and Economics (2018) 46:1–37 https://doi.org/10.1007/s10657-018-9589-4 



# **Resisting the extortion racket: an empirical analysis** 

### **Michele Battisti**<sup>**1**</sup> **· Andrea Mario Lavezzi**<sup>**1**</sup> **· Lucio Masserini**<sup>**2**</sup> **· Monica Pratesi**<sup>**2**</sup> 

Published online: 14 June 2018 

© Springer Science+Business Media, LLC, part of Springer Nature 2018 

### **Abstract** 

While the contributions on the organized crime and Mafia environments are many, there is a lack of empirical evidence on the firm’s decision to resist to extortion. Our case study is based on _Addiopizzo_ , an NGO that, from 2004, invites firms to refuse requests from the local Mafia and to join a public list of “non-payers”. The research is based on a dataset obtained linking the current administrative archives maintained by the chambers of commerce and the list updated by the NGO. The objective of this paper is twofold: first, to gather sound data on the characteristics of the _Addiopizzo_ joiners; second to model the probability to join _Addiopizzo_ by a two-level logistic regression model. We find that the resilience behavior is likely to be the result of both individual (firm) and environmental factors. In particular, we find that firm’s total assets, firm’s age and being in the construction sector are negatively correlated with the probability of joining AP, while a higher level of human capital embodied in the firm and a higher number of employees are positively correlated. Among the district-level variables, we find that the share of district’s population is negatively correlated with the probability to join, while a higher level of socio-economic development, including education levels, are positively correlated. 

**Keywords** Organized crime · Extortion · Social mobilization · Multilevel regression models 

**JEL Classification** O17 · K42 · R11 · C41 

> * Michele Battisti michele.battisti@unipa.it 

Andrea Mario Lavezzi mario.lavezzi@unipa.it 

Lucio Masserini lucio.masserini@unipi.it 

Monica Pratesi monica.pratesi@unipi.it 

> 1 Department of Law, University of Palermo, Piazza Bologni 8, 90134 Palermo, Italy 

> 2 Department of Economics and Management, University of Pisa, Via Ridolfi 10, 56124 Pisa, Italy 

Vol.:(0123456789)1 3 

European Journal of Law and Economics (2018) 46:1–37 

2 

## **1 Introduction** 

Organized crime poses a serious threat in several countries (Van Dijk 2007). Italy, in particular, is a relevant case in the European context, as powerful criminal organizations operate in some Italian regions such as Campania, Calabria and Sicily. Organized crime interferes with the economy in a number of ways (see Lavezzi 2014). For example it provides firms with _protection_ services, e.g. from competitors, from other criminal organizations, from police harassment; it promotes and enforces cartels for the adjudication of public works (Gambetta and Reuter 1995); it subtracts public resources from productive uses to private interests through its influence on local politicians (Fioroni et al. 2017). 

A typical activity of criminal organizations such as the Sicilian Mafia is extortion, consisting in the forced extraction of resources from firms, under the threat of punishment (see e.g. Paoli 2003). According to recent estimates, in cities like Palermo (Sicily) more than 80% of firms and stores pay the extortion racket (Confesercenti 2010 ). The impact of extortion on firms’ activity and on the aggregate economy can be sizable. For example, Balletta and Lavezzi (2016) estimate that for Sicilian firms the incidence of extortionary payments may reach 40% of gross profits, while Asmundo and Lisciandra (2008 ) find that the resources subtracted to the Sicilian economy by organized crime through extortion amount to 1.4% of regional GDP. 

Criminal organizations, therefore, can severely interfere with firms’ activity and hinder economic growth (Pinotti 2015). This makes the contrast to organized crime of paramount importance for economic policy in regions or countries where the phenomenon is widespread. A primary actor in the contrast of criminal organizations is obviously the State, but the civil society may also mobilize (see La Spina 2008 and Lavezzi 2014 ). In the case of extortion, in particular, firms may resist the racketeers and refuse to pay. 

The literature on extortion is abundant,<sup>1</sup> but little evidence has been analyzed so far on the decision of firms that resist extortion. To study this issue, in this paper we _Addi-_ focus on the unique experience of the firms of Palermo (Sicily) that joined _opizzo_ (AP), an NGO that, from 2004, invites firms to refuse extortionary requests from the local Mafia and join a public list of “non-payers”. AP originated from the idea of stimulating civic-minded consumers to buy products and services from AP2016) discusses firms, providing in this way incentives to firms to join the list. Dixit ( the bottom-up approach of AP as one of the most interesting recent experiences of civil society’s resistance to corruption and organized crime. 

The aim of this paper is to understand which type of firms are more likely to resist the extortion racket and join AP. To this purpose, we present a simple model of firms’ decision to join AP and then estimate a logit model to ascertain which observable firms’ characteristics display robust conditioned correlations with the 

> 1 See for instance Varese (2014), for discussion and references. Contributions in economics, covering both theoretical and empirical issues, include Alexander (1997), Konrad and Skaperdas (1998), Bueno de Mesquita and Hafer (2007), Asmundo and Lisciandra (2008) and Balletta and Lavezzi (2016). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

3 

decision to join, providing an interpretation of the results in the light of the theoretical model. In particular, we study the decision to join AP as a case-control study (Keogh and Cox 2014) on a unique dataset of AP-joiners and non-joiners, which includes data on firms and on the socio-economic characteristics of the administrative districts where the firms are located. Specifically, we adopt a multilevel logistic regression approach (see, e.g., Goldstein 2011; Raudenbush and Bryk 2002; Sneijder and Bosker 2011) in which firm’s and district’s characteristics respectively represent first- and second-level variables correlated with the decision of joining AP. 

Our main results are the following. We find that the probability to join AP is negatively correlated with the level of firm’s total assets and firm’s age, and positively correlated with the level of human capital embodied in the firm and with the number of employees. Belonging to the construction sectors also reduces the probability to join. We also find that second-level variables such as district’s population share and districts’ education level (as discussed for instance in Dixit 2016 ) have significant correlations with the decision to join AP (negative and positive, respectively). 

Understanding why some firms join AP has been previously addressed by scholars from other disciplines. For example, Vaccaro (2012) and Vaccaro and Palazzo (2015) adopt the perspective of management science and focus on the capacity of AP to convince firms to join. They find that AP investment in credibility as an organization and the strategic promotion of values such as dignity and legality are crucial factors. Their approach is however qualitative, and differs in this respect from the quantitative approach adopted in this paper. From the perspective of political science, Gunnarson (2014) instead focuses on the role of AP-members’ previously existing networks and on mutual trust in the decision to join.<sup>2</sup> Yet, no contribution so far provided a rigorous statistical analysis to identify which firm’s characteristics correlate with the decision to resist the extortion racket and join AP.<sup>3</sup> 

The paper is organized as follows: in Sect. 2 we describe the activity of AP; in Sect. 3 4 we discuss our theoretical assumptions on firms’ decision to join; in Sect. we introduce the research design, the data sources used in the study and the ways to recruit the firms. Section 5 contains the empirical analysis; Sect. 6 concludes and discusses some directions for further research. 

## **2  The case of** **_Addiopizzo_** 

_Addiopizzo_ means “farewell to _pizzo_ ”, where _pizzo_ is the Sicilian definition of the money extorted by _Cosa Nostra_ .<sup>4</sup> AP activity begun from an idea of few young activists who, in the night of June 29th, 2004, flooded the walls of Palermo with 

> 2 The experience of AP has been also studied as a peculiar example of “critical consumption” by Forno and Gunnarson (2010) and Partridge (2012). 

> 3 Gunnarson (2014) provides an econometric analysis of firms’ decision to join. The sample and the methodology are, however, very different from ours: the sample comes from a survey and does not include a control group of non-joiners and second-level variables. This limits the analysis to a study of the different _timings_ of the decision to join without, however, considering censored observations. 

> 4 Thorough accounts of the Sicilian Mafia are given by Gambetta (1993) and Paoli (2003). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

4 

thousands of stickers carrying the slogan: _“A whole people who pays the pizzo is a people without dignity”_ , with the aim of provoking a reaction of the civil society against the Mafia. This was a shocking message in a city where organized crime is historically rampant. 

In 2005 the founders of AP launched a campaign to spread this message of resistance to the racket which brought, in May 2006, to the creation of a list of more than 100 businesses available to publicly denounce the _pizzo_ , claiming their refusal to pay. The list was published in a local newspaper and was followed by diffusion in the national media. From 2004, more than 1000 firms have been members of AP and at the time we collected the data (May 2012), the number of joiners was around 820.<sup>5</sup> Occasionally, AP runs campaigns targeted to specific neighborhoods of Palermo and organizes meetings in schools. Also, it holds a regular event in May, _“Fiera del consumo critico”_ , in which AP firms present their products and various activities, from debates to live performances, take place.<sup>6</sup> 

In principle any kind of firm can join AP, i.e. there are no formal barriers to, e.g. firms belonging to a certain sector, of a certain size, etc. The number of AP joiners may fluctuate as new firms join and old firms are deleted from the list. Deletion from the AP list can occur for various reasons, for example: if the firm goes out of business, if it changes ownership and the new owners wants to be cancelled, if interactions with organized crime are detected by AP staff.<sup>7</sup> The last point calls into question the truthfulness of the firms’ commitment to leave the “pizzo system”. As pointed out by Vaccaro (2012, p. 7), firms can be “double-game” players and strategically choose to join an anti-racket organization to hide their actual connections with organized crime. AP, however, closely monitors the joiners and has already expelled some “double-game” players, although the number of cases seems very limited.<sup>8</sup> 

The idea of making public the list of joiners follows an economic insight. Consumers, in fact, are invited to shop at AP stores if they wish to: “pay those who do not pay”.<sup>9</sup> In other words, AP tries to elicit “critical consumption” by civic-minded 

> 5 These numbers include different branches of the same firm. The list can be consulted in the AP website (http://www.addio pizzo .org/). 

> 6 AP also provides legal support in trials against the racketeers, mainly in collaboration with the business association _Libero Futuro_ , or psychological support to entrepreneurs wishing to stop paying the Mafia. For more details on AP activity, see Forno and Gunnarson (2010, pp. 109–111) and Gunnarson (2014, pp. 42–44). 

7 Since we received the list of AP-firms as of May 2012, we cannot observe the relative weight of these causes on the fluctuations in the number of joiners in the period of observation. 

8 This was confirmed by personal communication from AP staff. We consider joining AP as signaling the non compliance with paying the Mafia. However, while joining AP is observable, not paying the Mafia is not. To join AP, firms’ owners must sign a declaration of non compliance with extortionary requests but we are not able to control for the firms’ truthful disclosure of information. As remarked, however, existing evidence suggests that cases of “double game” players are rare. On the other hand, whether firms that did not join AP are payers or not-payers is not observable. The available evidence (see Confesercenti 2010 and Asmundo and Lisciandra 2008) suggests that the large majority of firms in Palermo pay the _pizzo_ . Therefore, in this paper we posit that joining AP implies refusing to pay the _pizzo_ , while not joining implies paying. 

9 This is another slogan diffused by AP. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

5 

citizens who, in this way, can express their opposition to organized crime. AP stores clearly signal their membership by displaying an AP sticker at the entrance of their premises. This strategy may be seen as an attempt at eliciting positive consumer dis1957). crimination for AP-firms (Becker 

## **3  On the decision to join AP** 

We model the decision of firms to join AP as an economic decision based on: (1) the expected variation of economic profit from joining AP; (2) the expected cost of the decision, deriving from a possible punishment of “rebel” firms by the Mafia; (3) the antimafia values embodied in the firm, in particular in its owner(s). 

Specifically, we model this problem as a choice between alternatives providing different levels of utility. Denote by _Ui_ the expected flow of utilities of the owner of firm _i_ , when it is not a member of AP, at the time in which the decision to join can be made.<sup>10</sup> _Ui_ can be defined as: 



where _휋i_ indicates the expected flow of economic profits,<sup>11</sup> _xi_ denotes the _pizzo_ paid to the Mafia, _vi >_ 0 is a measure of firm owner’s antimafia values: the larger _vi_ , the higher the loss of utility of a firm staying in the _pizzo_ system;<sup>12</sup> _휖i_ is term capturing unobservable firm-specific characteristics affecting its utility when it is not a member of AP.<sup>13</sup> 

> Denote by _Ui_<sup>_ap_the expected flow of utilities of a firm that joins AP.</sup><sup>_U_</sup> _i_<sup>_ap_can be</sup> defined as: 



> where _휋_<sup>_ap_</sup> _i_<sup>indicates the expected flow of economic profits after joining AP,</sup><sup>_pi_is</sup> the probability of punishment by the Mafia, _Zi_ is the intensity of the punishment.<sup>14</sup> Clearly, a firm that joins AP stops paying _xi_ and does not suffer the loss of utility _휖_<sup>_ap_</sup> associated to the antimafia values of its owner(s). Finally, _i_<sup>is a term capturing</sup> 

> 10 For simplicity we assume that the utility function is linear and additive in its arguments, and omit the representation of the flow of utilities as an integral. 

> 11 Operating in the “pizzo system” may increase profits as, for example, the Mafia can influence the adjudication of public contracts to favor “protected” firms, or guarantee to such firms monopolistic power in local markets (see Gambetta and Reuter 1995 and Varese 2009). 

> 12 Clearly, firms owned by entrepreneurs expressing sympathy and approval for the Mafia, could be characterized by _vi <_ 0 . See Lavezzi (2014, pp. 177–78) for a discussion and references on anti-Mafia values in a society. 

> 13 In principle, anti-mafia values can also be unobservable. In the specification of Eq. (1) we keep them separated from the error term as they can be proxied by observable firms’ characteristics (see below). 

> 14the supply of offenses by a criminal. Factors _pi_ and _Zi_ correspond to those introduced by Becker (1968, p. 177) in his characterization of 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

6 

unobserved firm-specific factors affecting its utility when it joins AP. We assume that _휖i_ and _휖i_<sup>_ap_are IID type I extreme value.</sup> 

By subtracting Eqs. (1) from (2) it is possible to define the utility gain from joining AP as: 



where Π _i_ = _휋i_<sup>_ap_</sup> − _휋i_ and _휂i_ = _휖i_<sup>_ap_</sup> − _휖i_ . Equation (3) can be empirically studied by a logit model (see, e.g., Cameron and Trivedi 2005, p. 477). 

Equation (3), in particular, suggests that the probability to join AP is higher: the higher the expected increase in the economic profits, the higher the _pizzo_ , the lower the probability of punishment by the Mafia, the lower the intensity of the punishment, the stronger the antimafia values of firm’s owner(s). In our empirical analysis we will proxy for these factors by observable firms’ characteristics and by the Let us describe how characteristics of the districts where the firms are located.<sup>15</sup> to empirically proxy for the factors that, according to Eq. (3), affect the decision to join. 

First and foremost, profits increase after joining AP if consumers reward the firm by purchasing its goods and services.<sup>16</sup> Therefore, we expect that firms located in districts in which critical consumption is more likely to activate have a higher probability of joining AP. Given the absence of good measures of the propensity for critical consumption at district level,<sup>17</sup> we will proxy such propensity by measures of socio-economic development, in particular the demographic structure of the districts<sup>18</sup> and the education levels. Education is particularly relevant to our discussion as it represents an important channel through which individuals can acquire antimafia values. Schneider and Schneider (2003, Ch. 11) describe in details the spread in Sicily of the so-called “education for legality” through schools, especially at the primary level. Since the eighties, and in particular after the murders of judges Falcone and Borsellino in 1992, various programs have been launched to educate children to values such as loyalty, civicness, tolerance, trust and respect for gender differences, in opposition of the values of violence, vengeance, machism and _omerta’_ that characterize the culture of _mafiosi_ . 

> 15 The details on the dataset are provided in Sect. 4. 

> 16 In this paper we analyze the observable decision to join or not to join AP, and can therefore only speculate on firms’ expectations on the consequences of such decision. In a companion paper, Battisti et al. (2017a), we estimate the causal effect of joining AP on firms’ economic performance by adopting a propensity score matching technique. 

> 17 Partridge (2012, p. 16) considers the fraction of “critical consumers” that, as of 2010, registered in the Addiopizzo website, and finds a correlation with the share of AP firms by district of 0.8. However, the measure considered is biased by self-selection. The current version of the Addiopizzo website contains an updated list of consumers classified by zip code (and not by district). The lack of the date in which consumers registered, however, hides a crucial piece of information: i.e. whether they registered before or after firms in the district joined AP. 

> 18 Pedrini and Ferri (2014), for example, show that more educated and older individuals are more likely to become “responsible consumers”. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

7 

Furthermore, we will use measures of firms’ economic performance (revenues, gross and net profits, bank loans) as possible additional indicators of firm’s expected profitability from joining AP. For example, high-profit (or high-revenues) firms are less likely to join if their high profits (or revenues) come from their linkages with the Mafia. On the other hand, these firms can be more likely to join if high profits (or high revenues) make them less sensitive to possible negative consequences such as becoming credit-rationed by the banks, if the latter perceive AP-firms as more “risky”.<sup>19</sup> Let us remark that, while the theoretical model suggests that the decision to join be based on the expected difference in the flow of profits accruing to the firm before and after joining AP, in the empirical analysis of this paper we test the capacity of indicators of firms’ profitability _at the time_ the decision to join is taken to proxy for this expectation. In other words, being _휋i_ the expected profit flow after joining we can check whether it could be proxied by the observed profit flow at the time the decision of joining is made.<sup>20</sup> 

As regards the level of the _pizzo_ paid by the firms, unfortunately no data are available for the sample we analyze. However, in an analysis of extortion on a sample 2016 of Sicilian firms, Balletta and Lavezzi ( ) find that Mafia “taxation” is highly regressive, as the share of _pizzo_ on profits is much larger for smaller firms.<sup>21</sup> Therefore, we conjecture that larger firms have a lower propensity to join AP, and will consider the possible effect of _xi_ through a measures firms’ size such as the capital stock. Yet, the size of the firm can also proxy for its monopolistic power. As long as 2014 such monopolistic power is granted by the Mafia (see Lavezzi ), larger firms will be less prone to join AP, as this would imply giving up the monopolistic rents and reduce their profits.<sup>22</sup> 

Refusing to comply with Mafia rules may imply punishment with a certain probability and intensity on the part of the mobsters. A well-known characteristic of the “Mafia trademark” is, in fact, the use of violence (Gambetta 2009).<sup>23</sup> In other words, risk is certainly a crucial factor that firms consider in their decision to join AP. Some variables in our dataset allow us to consider this aspect. 

> 19 We became aware of this possibility by personal communication from AP staff. A similar reasoning to the one based on profits or revenues can be done with respect to other measures of firms’ economic “health”, such as the loans/revenues ratio. 

> 20 We defer to companion papers a deeper analysis of this crucial aspect. See Battisti et al. (2017a) and Battisti et al. (2017b). 

> 21 The sample analyzed by Balletta and Lavezzi (2016) includes 120 firms that paid _pizzo_ between 1991 and 2006. 

22 We thank an anonymous referee for suggesting us this possibility. Recent literature (e.g. Mete 2011) highlights that large firms may establish economic deals with criminal organizations taking advantage of their bargaining power, for example in the adjudication of public works. In this case large firms are not victims of extortion and, therefore, might have lower incentives in cutting the ties with the Mafia and join an NGO such as _Addiopizzo_ . 

23 There exists recent evidence of intimidation to AP firms, reported in the local news (see, e.g., Fiasconaro 2007, Ziniti 2015 and MNews (2015)). There is, however, also evidence contrary to this assumption, according to which _mafiosi_ are unwilling to retaliate AP-firms, because this would attract attention by the police (see, e.g, the declaration of M. Pasta in the news section of http://www.addio pizzo .org/). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

8 

Our dataset contains the number of employees in a firm. We will use this as a measure of the perceived probability of being punished through personal harm, on the part of the firm’s owner(s). That is, assuming a certain probability of retaliation against individuals owing or working in a firm, the probability of being punished is decreasing in the number of employees. In addition, since age can be negatively correlated with risk-taking (see, e.g. Vroom and Pahl 1971), we will consider firm’s age as a covariate.<sup>24</sup> Finally, we will exploit information on firms’ sector as sectors may display different degrees of penetration by organized crime. A notable example is the construction sector, which is heavily controlled by the Mafia (see e.g. Lavezzi 2008). Firms in the construction sector, therefore, are likely to perceive a higher probability of punishment if they join AP or, as long as they can benefit from Mafia protection in the adjudication of public contracts, they might expect a loss of profits if they leave the _pizzo_ system. 

As regards the intensity of the punishment, _Zi_ , we suggest that it can be proxied the value of the assets of the firm, under the hypothesis that another likely form of punishment takes the form of damaging the firm’s physical infrastructure (see, e.g, Fiasconaro 2007, Ziniti 2015). In this case, we conjecture that a higher level of firm’s assets reduces the probability of joining AP. 

Finally, as remarked above, the literature suggests that anti-Mafia values are likely to be correlated with education. Since owner’s education is unobservable we will consider the human capital embodied in the firm as a proxy for the unobservable human capital of owners. 

## **4  The data sources** 

The data sources for this study are three: the list of AP joiners from 2004 to 2012, maintained by the AP organization, the list of active companies updated and maintained by the chambers of commerce of the city of Palermo in the same period, and the socio-economic and demographic data on the districts of Palermo from the population census 2001. Our cross-sectional study refers to firms in the AP list as of May 2012. 

In the list of AP joiners we only consider joint-stock companies because balance sheets, a relevant source of information for our analysis, are available from 

> 24 Firms’ age will proxy for the unobservable owners’ age. Unfortunately, we have information on the age of AP firms owners (i.e. of the person who signed the form to join AP) for a subset of 94 observations, while this data is not available for the control group. The correlation with firm’s age equals 0.33 when all observations are included, but increases to 0.58 when 8 outliers are removed. The average age of the AP-firm owners of this sample (in 2005) is 41. In addition, firm’s age is a measure of the length of time in which the firm operated in a Mafia-infested environment. It is reasonable to assume that the longer the period, the stronger the relationship it may have developed with the racketeers, the harder is breaking these relationships. Also for this reason, therefore, we expect a negative effect of firm’s age on the propensity to join AP. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

9 

the archive of the chambers of commerce only for this type of firm. In addition, we excluded firms’ branches, as the balance sheet data only refer to the firm’s headquarter. Finally, we excluded observations on 54 firms outside Palermo, as they are located in small towns, where the district characteristics’ variation cannot be observed as in Palermo. By this procedure, we identified 150 joint-stock companies located within the municipality of Palermo. The linkage of these firms with the register of the chambers of commerce (CCIAA) provided data on balance sheets for the period 2002–2011, to have approximately 5 years of data before and after the creation of AP. 

The other joint-stock companies active in Palermo in May 2012 were listed in the CCIAA archive. Also for them the register provided the balance sheets for the period 2002–2011. Unfortunately the CCIAA gave us the access to a research sample of 483 out of the 9400 joint-stock companies active in the province. The simple random sample was stratified by age of the firm since the beginning of its activity. This to resemble the age distribution of the AP joiners: 72% of the firms existed before December 31st, 2005, and 28% were companies created after that date. We enriched the list of AP joiners with auxiliary information from Census data on the 25 Palermo districts where the firms are located. The linkage between the two sources has been done by geographical matching of the address of each joiner with the database from the 2001 Census, containing data on the socio-economic conditions of each district.<sup>25</sup> The choice of this Census year is the closer with respect to the decision to join AP, which appeared only in 2004. 

Finally, we built a dataset consisting of two types of data for each firm: firm-specific or individual data, and district-specific data on the socio-economic characteristics of the district were the firm is located. Our final dataset, as mentioned, includes 150 AP-joiners firms and 483 firms not AP-joiners.<sup>26</sup> 

The dataset contains firm-level and district-level data. Firm-level data include: sector, age, date of joining AP, capital stock, personnel costs, revenues, gross profits, net profits, debts/revenues ratio, number of employees.<sup>27</sup> For each firm with at least three available observations, we averaged the yearly values over the period 

> 25 Census data are collected by ISTAT, the Italian National Institute of Statistics. Originally, the Census dataset contained data on 3021 census cells, from which we computed values for 25 districts, after matching each cell to a district. 

> 26 Overall, we are considering a sample of 633 firms that have been operative in the period 2002–2011. The average number of joint-stock firms operating in the same period is around 9400, including both “active” and “inactive” firms (i.e. firms that have not yet started their activity or failed to communicate to the CCIAA the beginning of activities). Our sample, therefore, covers approximately a range of 7–8% of the population. 

> 27 Nominal data were converted into real terms by the consumption price index (CPI) of Palermo, from the Istat System of Territorial Indicators (SITIS). All data are expressed at constant 2000 prices. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

10 

2001–2012 for each balance sheet variable to reduce the impact of cyclical components, obtaining a cross-sectional database. The firm age is computed between the initial year of firm’s activity and 2012. 

District-level data are from the 2001 census. In particular we selected indicators of socio-economic development such as: 

1. Demographic characteristics: district’s population share, dependency ratio (share of population above 64 on share of population under 15), share of small families (with 1–3 components), share of large families (with 5–6 components). 

2. Human capital levels in the form of education: share of population with tertiary, secondary (high school), primary (including junior high), elementary education; share of literate population,<sup>28</sup> share of illiterate population.<sup>29</sup> 

3. Labour market conditions: employment and unemployment rates, employment share in agriculture, manufacturing, construction, services; shares of selfemployed and employees on total labour force. 

4. Housing conditions: share of houses in good/perfect conditions, share of houses with no running water. 

Table 1 contains average values and standard deviations of the firm-specific variables from the balance sheets, along with the _p_ values of _t_ test for the difference of the means, while Table 3 in “Appendix A” contains their correlations. 

Table 1 shows that, AP-firms have significantly higher (at 5%) levels of gross profits and significantly lower loans/revenue ratios<sup>30</sup> and firms’ age, while Personnel costs and number of employees are significantly higher at 10% significance level.<sup>31</sup> 

“Appendix B” contains the description of the 25 Palermo districts, highlights the distribution of AP-firms in the districts and some examples of the spatial distribution of the indicators of socio-economic development. Overall, Palermo appears as a city in which two different socio-economic contexts co-exist: one in which population is on average educated, generally employed and lives in families of small size, and one in which population has on average little or no education, lives in large families, and is characterized by high unemployment rates. In the following section we describe our strategy for the econometric analysis. 

> 28 This category includes people with no schooling, but able to read and write. 

> 29 The population considered to compute these shares includes individuals aged more than 6 years. 

> 30 The value of the loans/revenues is obtained after cancelling few extremes observations. This variable is very sensitive to this problem, that we will take into account in the econometric analysis. 

> 31 The average number of employees of Italian firms in 2005 was 12.6 and 3.7 for, respectively, joint stock companies and partnerships (Istat 2007). From this perspective, therefore, it can be noted that the AP (control) firms have a higher-than-average (lower-than-average) figure. Since the current analysis focuses on joint-stock companies, it is possible that we are observing a sample of somewhat larger firms (the number of employees in the official statistics is the criterium utilized to measure firms’s size). However, in the population of joint-stock companies in Italy and in Sicily, there exists a relevant share of firms with a number of employees belonging to the class 0–9. In particular, in 2015 this share was respectively for Italy and for the province of Palermo, 83 and 86% (source: http://dati.istat .it/. These figures are not available for previous years). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

11 

## **5  Empirical analysis** 

In this section we describe the methodology we adopted to identify which variables are associated to the decision of joining AP, and then present the results. 

### **5.1  Econometric analysis: modeling the participation to AP** 

Hierarchical or multilevel data are common in the social and behavioral sciences. In this paper, analyzed data show a typical hierarchical structure (Raudenbush and Bryk 2002; Snijders and Bosker 2011) where lower-level units (individuals) are nested within higher-level units (clusters). Here, firms are clustered in districts and define a two-level hierarchical data structure. 

Because of this we can expect that firms located within each district, sharing the same unobserved factors due to the exposure to common environmental or contextual effects, have correlated values of the response variable. When this occurs, analyzing lower-level units as if they were independent can produce biased standard errors of the regression coefficients, thus resulting in erroneous inferences (Hox 2010 ), and possible substantive mistakes when interpreting the effects of predictor variables. Furthermore, in the analysis of such data, it is usually informative to take into account the sources of variability in the responses associated with each level of nesting, in this case the variance between firms and between districts, respectively. 

Multilevel regression models (Goldstein 2011; Raudenbush and Bryk 2002; Sneijder and Bosker 2011) are suitable for handling dependence among the responses resulting from a hierarchical data structure, also analyzing the complex pattern of variability. In multilevel models, the total variance of the response variable is partitioned into its different components of variation, due to the various cluster levels in the data. The effect of clustering is modeled by introducing random effects (Laird and Ware 1982), that is a continuous latent variable following a known parametric distribution, whose values are constant within clusters but vary across clusters. Independence across observations is assumed at cluster-level (district) whereas at individual-level (firms) it is assumed only among units belonging to different clusters (independence conditional on cluster membership). As a consequence, clusterlevel random effects can be interpreted as the effects of district-levels unmeasured covariates that induce dependence among firms in the same neighborhood whereas individual-level random effects represent residuals specific to each firm after taking into account cluster effects. To explain at least some of the cluster-level variability, district-level covariates can also be introduced. 

In this research, the response variable, indicated with _y_ , is binary and distinguishes the firms that decided to join _Addiopizzo_ by May, 2012 ( _y_ = 1 ) from the others ( _y_ = 0 ). Following the model of Sect. 3, the analysis is performed by using a two-level random intercepts logistic regression model ( Goldstein 2011; Rabe-Hesketh et al. 2004). Given a binary outcome _yij_ [0, 1] observed on firm _i_ , with _i_ = 1, 2, … , _Nj_ , located in neighborhood _j_ , with _j_ = 1, 2, … , _G_ , and being _Pij_ = _Pr_ ( _yij_ = 1) the probability that _yij_ takes on the value of 1 (i.e. the firm is 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

12 

**Table 1** Firm-specific variables in cases and control groups 

||AP||Control||_p_value|
|---|---|---|---|---|---|
||Mean|SD|Mean|SD||
|Total assets|748,178.32|4,709,943.16|615,897.62|3,380,364.93|0.75|
|Personnel costs|346,859.69|1,549,325.66|122,998.12|404,548.88|0.08|
|Number of employees|15.32|60.23|6.06|17.39|0.07|
|Loans|463,637.72|2,652,589.76|428,064.21|2,149,966.68|0.88|
|Revenues|1,477,937.55|5,086,066.25|909,457.68|3,643,809.14|0.21|
|Loans/revenues<br>|0.39|2.1|1.81|13.10|0.03|
|Gross profts<br>|41,947.74|178,315.01|9,915.70|169,033.71|0.05|
|Net profts|− 2,812.01|81,004.82|−  20,138.31|232,021.24|0.16|
|Firm’s age (years)|11.90|12.60|15.00|11.22|0.01|



associated with _Addiopizzo_ ), the model is defined in terms of the natural logarithm of the odds ratio (logit), indicated as _ln_ ( _Pij_ ∕1 − _Pij_ ). 

Hence, the two-level random intercepts logistic regression model can be expressed as a linear function of the explanatory variables using the single equation mixed model formulation (Rabe-Hesketh et al. 2004): 



where _xij_ is a vector of predictors for firm _i_ placed in district _j_ and _wj_ is a vector of predictors characterizing district _j_ . 

The random effects are given by the level-2 residuals, _uj_ ∼ _N_ (0, _휎u_ ) , which define the effect of being in district _j_ on the log-odds. This parameter represents a continuous and unobservable quantity shared by the firms within a particular neighborhood that captures all the relevant factors not accounted for by the observed covariates. The magnitude of the standard deviation, _휎u_ , indicates the strength of the influence of the specific district _j_ on the log-odds. 

The fixed parameters to be estimated are _훽_ 0 , _훽_ 1 and _훾_ . More specifically, _훽_ 0 represents the population average log-odds when _xij_ = 0 and _uj_ = 0; _훽_ 1 is the vector of the _x_ for regression coefficients quantifying the effect on log-odds of a 1-unit increase in all the firms in the same neighborhood, thus having the same value of _u_ ; _훾_ is the vector of the regression coefficients for the predictors characterizing the neighborhood _j_ . The probability of joining AP for firm _i_ in neighborhood _j_ is calculated as follows, for given values of the predictors _xij_ , _wj_ and _uj_ , the specific term: 



1 3 

European Journal of Law and Economics (2018) 46:1–37 

13 

From the previous formula it is also possible to make predictions for “ideal” or “typical” firms having particular values for the vector of covariates, given the value of the random effect. The measurement of the extent to which the observations in a cluster are correlated is often of interest and can be expressed by the intraclass correlation coefficient (ICC), indicated with _휌_ . This quantity can be obtained as the ratio of the variance of the random effects _uj_ to the total variance and can be interpreted as the proportion of variance explained by clustering. Since the logistic distribution for the level-one residual implies a variance of _휋_<sup>2</sup> ∕3 , the intraclass correlation in a two-level logistic random intercept model is defined as follows: 



This formulation can be used also to express the residual intraclass correlation coefficient, that is the intraclass correlation after controlling for the effects of the explanatory variables. Methods for estimating hierarchical or multilevel logistic models are based on maximum likelihood (ML) (Demidenko 2004; Skrondal and Rabe-Hesketh 2004; Tuerlinckx et al. 2006) or, alternatively, on Bayesian methods (Browne and Draper 2000; Congdon 2006; Draper 2008). 

When the second level effects are treated as random and the model parameters as fixed, inference is usually based on the marginal likelihood, that is the likelihood of data given the random effects, integrated over the random effects distribution. In this case, except for multilevel linear models, parameter estimation involves inevitably numerical methods and some kinds of approximation because the integrals do not have a closed-form solution (Pinheiro and Bates 1995; Skrondal and RabeHesketh 2004). Currently the most used algorithms for approximating the integral employed in the calculation of the log likelihood are the Laplace approximation and adaptive numerical quadrature. In contrast, when both the random effects and the model parameters are treated as random variables, a Bayesian approach is applied and inference is based on the posterior distribution, given the observed data. Bayesian methods use Markov chain Monte Carlo (MCMC) simulation methods for sampling from the posterior distribution and estimating parameters by their posteriors means (Gelfand and Smith 1990; Clayton 1996). 

In this paper, ML approach is employed and estimation of the model parameters is performed using the _melogit_ procedure, implemented in the software Stata 13.0. The integral required to calculate the log-likelihood is approximated by using the mean-variance adaptive Gauss-Hermite quadrature (Skrondal and Rabe-Hesketh 2004) with 20 points of integration (with 20-point adaptive quadrature). Following this approach, the quadrature locations and weights for individual clusters are updated during the optimization process by using the posterior mean and the posterior standard deviation. 

Prediction of random effects and expected responses is also often required. An extensive treatment of this topic is addressed by Skrondal and Rabe-Hesketh (2004, 2009 ). For assigning values to random effects, empirical Bayes prediction (Efron and Morris 1973, 1975; Morris 1983; Maritz and Lwin 1989; Carlin and Louis 2000a, b) is employed. For this method, Skrondal and Rabe-Hesketh (2009) also discuss three 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

14 

different kinds of standard errors (the posterior standard deviation, the marginal prediction error standard deviation and the marginal sampling standard deviation). 

### **5.2  Econometric analysis: results** 

The analysis is carried out by using a two-level logistic regression model, estimated on 633 firms nested within 25 districts, the average number of firms for each district is 23.5, with a minimum of 2 and a maximum of 148. 

Our modeling strategy consists in comparing the estimates of different specifications (Table 2 ), moving from a model with the first-level covariates only (Model 1), to the alternatives ones which include also the second-level (or district-level) covariates (Models 2–5), introduced to take the possible effects of contextual or environmental predictors into account.<sup>32</sup> 

Model 1 includes all the first-level variables only. The model estimates indicate that the variables proxying for firms’ performance (Revenues, Gross and Net Profits, Debt/Revenues ratio) are not significant. Total assets have a negative and significant effect, as well as firms’ age. Personnel costs and the dummies for employment level have a positive and significant effect, while the dummy variable for the construction sector has a negative and significant effect. 

Before introducing the second level variables, we find that the random intercept _휎_<sup>2</sup> variance is relevant and significant ( _u_<sup>= 0.2930;</sup><sup>_p_value</sup><sup>_<_0.0388).33 This indicates</sup> the presence of unobserved between-districts heterogeneity, meaning that the probability of joining _Addiopizzo_ is significantly different across the districts of Palermo, after taking into account model’s firm-specific covariates. More specifically, the random intercept parameter can be thought of as the combined effect of omitted firmspecific covariates that cause the firms within the same district to be more or less prone to join _Addiopizzo_ (Rabe-Hesketh and Skrondal 2006). The within-district 

32 Quantitative variables having different magnitude (Total assets, personnel costs, loans, revenues, gross and net profits, loans/revenues ratio) were preventively standardized in order to make the effects more easily comparable. To measure the number of employees, we did not use the absolute values of employees declared by the firms as it is often not precise due to the diffusion of illegal labour. In addition, this number is not part of the yearly balance sheets, but it is recorded in a given year and not regularly updated. Therefore, the variable “Number of employees” was transformed into categorical with three levels: “No employees”, indicating firms that utilize self-employed workers; “1–9 employees” and “more than 9 employees”. This classification reflects the one utilized in the Italian official statistics. Data from Movimprese (https ://www.infoc amere .it/movim prese ) show that in 2015 the share of firms in Italy and in Palermo with less than 10 workers is around 85% of the total number of active firms, so our categories allow focusing on two bins that represent respectively 85 and 15% of population, instead of looking to several class of firms with insignificant shares. Finally, the classification of economic sectors was considered at the most aggregate level, by distinguishing firms into three levels: “Manufacturing/Energy”, “construction” and “services”. 

> 33 The reported _p_ value is based on the likelihood-ratio (LR) test but it should be noted that the null hypothesis for this test is on the boundary of the parameter space because it refers to a variance component. As a consequence, the LR test does not have the usual central chi-square distribution with one degree of freedom but it is better approximated as a 50:50 mixture of central chi-squares with zero and one degree of freedom (Snijders and Bosker 2011). In Table 2 we report the significance level of the district variance for all models. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

15 

dependence among the dichotomous responses can be quantified by the conditional intraclass correlation (or residual intraclass correlation), indicated with _휌_ . The estimated value is _휌_ = 0.082 , and indicates the presence of a substantial association in the probability of joining _Addiopizzo_ among the firms within each district. These preliminary results confirm that a two-level approach is suitable for analyzing such data. The specifications in Models 2–5 introduce contextual or environmental predictors in order to account some of the second-level variance. In particular, we first of all tried to introduce each second-level variables individually. Then we tried all possible combinations holding fixed the significant ones and adding another variable. In Table 2 we only report the specifications with significant second-level covariates’ effects.<sup>34</sup> 

Models 2 and 3 feature the introduction of variables on the demographic characteristics of the districts. The population share has a negative and significant effect, In Model while the share of small families has a positive and significant effect.<sup>35</sup> 3, however, significant district variance remains. Model 4 adds to first-level variables a measure of the labour market characteristics. It shows that the share of selfemployed has a positive and significant effect.<sup>36</sup> 

In Model 5 we introduce a measure of districts’ human capital levels, controlling for the district population share. When we introduce the human capital variables alone, we find non-significant effects. Differently, controlling for the population share, we find that the share of population with elementary education has a negative and significant effect. In particular the significant education variable refers to primary education including elementary education only, although similar results are obtained with the share of individuals with primary education including elementary and intermediate.<sup>37</sup> 

In the next section we propose an interpretation of the results, in the light of the theoretical model of Sect. 3.<sup>38</sup> 

### **5.3  Interpretation of results** 

Results in Table 2 allow for the identification of which firm-level characteristics are significantly correlated with the decision to join AP. In addition, they demonstrate that the characteristics of the district where the firm is located have significant explanatory power. 

> 34 In particular, in some cases the estimation did not reach convergence, in all the others the estimated coefficient of the second variable were not significant. For this reason we did not try specifications with three second-level variables. 

> 35 We found that, when introduced individually, the effects of the share of large families and of the dependency ratio are not significant. 36 All the other labour market variables have non-significant effects. 

> 37 Results are available upon request. The two measures of primary education are strongly correlated (see Table 8). The consideration of other education measures returns non-significant results. 

> 38 Table 10 in “Appendix C” we contains the results of a test for the stability of the coefficients of the estimated models of Table 2. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

16 

**Table 2** Mixed effect logistic regression 

||(1)|(2)|(3)|(4)|(5)|
|---|---|---|---|---|---|
|Constant|− 1.601***|− 1.438***|− 1.681***|− 1.666***|− 1.442***|
||(0.510)|(0.506)|(0.508)|(0.501)|(0.491)|
|Capital stock|− 0.953*|− 0.998**|− 0.976*|− 0.963*|− 1.021**|
||(0.493)|(0.495)|(0.499)|(0.493)|(0.494)|
|Personnel costs|0.964**|1.005**|0.974**|0.964**|1.012**|
||(0.439)|(0.441)|(0.444)|(0.440)|(0.441)|
|Revenues|− 0.209|− 0.215|− 0.185|− 0.202|− 0.201|
||(0.239)|(0.240)|(0.236)|(0.239)|(0.233)|
|Net profts|− 0.0409|− 0.0464|− 0.00303|− 0.0142|− 0.0490|
||(0.431)|(0.417)|(0.438)|(0.430)|(0.394)|
|Gross profts|0.321|0.326|0.313|0.327|0.345|
||(0.381)|(0.383)|(0.383)|(0.378)|(0.374)|
|Loans/revenues ratio|0.0343|0.0355|0.0338|0.0342|0.0353|
||(0.0294)|(0.0300)|(0.0296)|(0.0298)|(0.0294)|
|Firm age|− 0.0274**|− 0.0270**|− 0.0272**|− 0.0274**|− 0.0268**|
||(0.0117)|(0.0117)|(0.0116)|(0.0116)|(0.0117)|
|Employee Class I|1.026***|1.012***|1.033***|1.017***|0.998***|
||(0.307)|(0.306)|(0.307)|(0.306)|(0.304)|
|Employee Class II|1.800***|1.797***|1.817***|1.812***|1.769***|
||(0.370)|(0.370)|(0.370)|(0.368)|(0.369)|
|Construction dummy|− 1.066**|− 1.123**|− 1.115**|− 1.124**|− 1.183**|
||(0.525)|(0.523)|(0.524)|(0.523)|(0.519)|
|Services dummy|− 0.302|− 0.331|− 0.369|− 0.369|− 0.372|
||(0.415)|(0.412)|(0.413)|(0.411)|(0.411)|
|Population share||− 0.360**|||− 0.456***|
|||(0.171)|||(0.153)|
|Small families share|||0.313*<br>(0.182)|||
|Self-employed share||||0.298**<br>(0.151)||
|Elementary education share|||||− 0.239*<br>(0.128)|
|District variance|0.293**|0.158|0.237*|0.155|0.0185|
||(0.276)|(0.203)|(0.252)|(0.216)|(0.117)|
|N|558|558|558|558|558|



Dependent variable: probability to join AP. Standard errors in parentheses. * _p <_ 0.10, ** _p <_ 0.05, *** _p <_ 0.01 Standard errors in parentheses. * _p <_ 0.10 , ** _p <_ 0.05 , *** _p <_ 0.010 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

17 

In particular, the negative effect on the probability to join of the capital stock is consistent with the hypothesis that larger firms have lower incentives to join because they experience a lower burden from payments of _pizzo_ (Balletta and Lavezzi 2016), and with a risk-minimizing behavior on the part of the firm, as the installed capital can be an easy target for organized crime.<sup>39</sup> However, it cannot be ruled out that larger firms may enjoy competition-reducing services by the mafia or may actually be partners of criminal organizations, and are therefore less willing to join NGOs such as AP.<sup>40</sup> 

Personnel costs, on the contrary, have a positive effect. As remarked, we consider this variable as a proxy for the human capital embodied in the firm and correlated with the human capital of owners.<sup>41</sup> The positive effect shown in Table 2, suggests that a firm embodying a higher level of human capital has a higher probability to join AP, as this may reflect the anti-Mafia values embodied in the firm. We interpret the positive effect of the number of employees as suggesting that the perceived risk is lower if more individuals are involved, assuming that in firms with more employees the risk of retaliation is shared among a larger pool of possible targets.<sup>42</sup> The negative effect of firm’s age follows our conjecture according to which younger firms are run by young, risk-prone, owners, who have weaker connections with the Mafia. 

Finally, the negative effect of the dummy variable for the construction sector is consistent with the hypothesis that the probability to join is lower in sectors under 1995), crimi- tight control by the Mafia. As pointed out by Gambetta and Reuter ( nal organizations enforce cartels of firms to adjudicate public contracts, which often involves firms from the construction sector. Our result suggest that firms in this sector, indeed, might find less attractive than firms in other sectors leaving the “Mafia system” as this might reduce their benefits. 

Other firm-specific characteristics such as profits, revenues, loans/revenues ratios turned out to have a non-significant effects on the decision to join. The theoretical model of Sect. 3 suggests that the decision to join should be based on the expected profitability of the choice. In the current setting, however, we cannot observe the 

39 In the cases described in Fiasconaro (2007) and MNews (2015), for example, the premises of the two firms were damaged by, respectively, arson and shots in the windows. 40 The exploration of the negative effect of firms’ size on the probability to join AP is an interesting topic for further research. In particular, this would require data on local indices of firms’ concentration in Palermo (currently not available), and information on whether existing large firms not belonging to AP owe the increase in their size to the protection from competition guaranteed by the Mafia or, as suggested by Mete (2011), they formed partnerships with the Mafia exploiting the bargaining power given by their size. 

41 The overall amount of personnel costs can proxy for the amount of human capital embodied in a firm since we control for the number of workers. A more suitable measure of firm’s human capital would be the average personnel cost. However, given the low reliability of data on the numbers of employees (see Footnote 32), we did not utilize such measure. 

42 An alternative useful measure would be the number of owners, assuming that they could be considered as responsible for the choice of joining AP, but this number is not observable. The largest majority of firms with 0 employees, i.e. firms utilizing self-employed workers, use a very low number of workers. According to Istat (2007), in 2005 the percentages of such firms employing respectively 1, 2, or 3 workers was 84, 13 and 3%. Joining the categories of firms with 0 and 1–9 workers does not affect our results. Results are available upon request. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

18 

expectations of the firms before the choice is made.<sup>43</sup> The results suggest that these variables are not good indicators of the expected profitability. 

The results on the second-level variables suggest that in highly populated districts the probability to join AP is lower and that, controlling for population, a high fraction of population with primary education reduces such probability. Table 5 shows that a high population share is more likely to be found peripheral districts such as Settecannoli, Villagrazia-Falsomiele, Mezzomonreale and Oreto-Stazione, although the central district of Liberta’, populated by high/middle class families, has a comparable population share. The former districts, however, have a share of population with primary education which is much higher (around 30%) than in the Liberta’ district, where it amounts to 11%. Results of Models 2 and 5 indicate that the sheer number of inhabitants is not a stimulating factor for firms to join AP, a fact that we interpret as suggesting that the presence of a large number of consumers is not a sufficient condition for firms to expect a significant level of “critical” consumption. However, the differences in the lowest levels of education matter the most to stimulate the decision to join AP. The fact that differences in the levels of other measures of education do not significantly correlate with the decision to join, suggests that the primary education level in the district is particularly relevant for this particular form of anti-mafia behavior. 

In general, a higher level of socio-economic development appears associated to the decision firms’ decision to join AP. Models 3 and 4 suggest that the presence of a large share of small families, or of self-employed workers (e.g. professionals), can be a good proxies for this.<sup>44</sup> 

We may try to give a quantitative assessment of the results by computing the odds ratios, following the specification of Model 5.<sup>45</sup> 

For a standard deviation change in the capital stock (3.898.049€), the odds of joining AP are expected to vary by a factor of 0.360, holding all other variables constant; instead, by taking a change of 100.000€ as a more typical value, the corresponding variation in odds is 0.974, which means that they are reduced by 2.8%. 

> 43 In a companion paper, Battisti et al. (2017b), we will explore the _actual_ profitability of the choice by comparing indicators of economic performance, such as profits, before and after joining AP. 44 Second-level variables can also explain the location choice of the firms. An analysis of this issue goes beyond the scope of this paper and will be carried out in Battisti et al. (2017b). 45 Since the research design is based on a retrospective unmatched case-control study, sampling of firms is performed conditional on the outcome variable with the consequence that the probabilities of joining AP are determined by the sample design. Accordingly, the baseline probability in the population is different from the corresponding proportion in the sample and interpreting the effect of independent variables in terms of the effects on the probability of being a case versus being a control has no substantive meaning. In such situations, odds ratios may provide the best alternative for interpretation since their values are invariant under study design (Agresti 2002; Hosmer et al. 2013; Keogh and Cox 2014). Odds ratio are obtained by taking the exponent of the regression coefficient, OR = exp( _훽_ ), and represents the factor of expected change in the odds of joining AP, holding all other variables constant. The relevant null hypothesis for odds ratios usually is _H_ 0 : OR = 1, and this corresponds directly to the null hypothesis that the corresponding regression coefficient is zero, _H_ 0 : _훽_ = 0. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

19 

Regarding Personnel Costs, a standard deviation change (888.085.9€) translates to an increase of odds by a factor of 2.751, whereas a change of 100.000€ implies a factor of 1.121, equal to an increase of 12.1%. Moreover, for firms in the construction sector, the odds decrease by a factor of 0.306 compared with firms in Manufacturing/Energy, resulting in a reduction of 69.4%. Also, for each additional year of firms’ activity, the odds reduce of about 2.6%. Finally, odds tend to increase with the number of employees: indeed, firms with 1–9 employees are more likely to join AP than firms with no employees, with the odds increasing by a factor of 2.713, whereas for firms with more than 9 employees the odds increase even more, by 5.865. Regarding the district-level variables, for a change of 0.01 in the districts’ population share, the odds of joining AP are expected to vary by a factor of 0.634, corresponding to a reduction of 36.6%, whereas the same change in the share of population with elementary education produces a variation of odds by a factor of 0.787, which reflects a decrease of 21.3%.<sup>46</sup> 

## **6  Conclusions** 

In this paper we studied the firms that resist the extortion racket in areas where organized crime is pervasive. Starting from the unique experience of _Addiopizzo_ we built a database of firms’ and districts’ characteristics for the city of Palermo in the period 2001–2012. We represented the decision to join AP by a simple economic model, and then identified the variables that are significantly correlated with the probability to join, highlighting that they refer to both firms’ and districts’ characteristics. 

These results shed light on the features of the firms that can create a bottom-up movement against organized crime (Dixit 2016). Moreover, an important suggestion coming from our result is that, confirming previous insights, promoting education (in particular by reducing the share of population with elementary education only) can represent an anti-Mafia policy option. In particular we identified a new channel: education and the related possibility of acquiring anti-Mafia values may imply a higher propensity of citizens to support firms that eventually decide to resist the extortion racket. 

Our analysis is the first of this kind and possesses obvious limitations. For example, the observable characteristics of firms and districts we utilized often represent proxies for some factors that we deem relevant. Further research based on actual 

> 46 We also checked for the the possibility of spatial spillover in the decision of joining AP. We computed the univariate Moran index, based on rook contiguity, and found a value of spatial correlation of 0.31 by considering the absolute numbers of joiners. This value, however, drops to − 0.04 for the number of joiners is weighted by the number of firms in the district. So the decision to join may be driven by some spatial spillovers but it is unclear how strong this pattern is. This aspect represents an interesting topic for future research. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

20 

measures will provide more precise answers to the issues raised in this paper. In addition, an important aspect that we did not address in this article is the interaction of firms’ decisions. It is likely that the decision to join AP is influenced by the number of firms that took the same decision before. In this light, an organization such as AP acts as a coordination device that assist firms in making such choices. This issue is tackled in Battisti et al. (2017b), where the tools of the social interaction econometrics are employed. 

**Acknowledgements** We are grateful to _Addiopizzo_ for providing data and for support to this research. We wish to thank in particular Caterina Alfano, Daniele Marannano, Giuseppe Pecoraro and Pico di Trapani. We also thank the Chamber of Commerce of Palermo, the Istat office of Palermo (in particular Dott. Li Vecchi), the _Ufficio Toponomastica_ of Palermo City Council (in particular Dott. Salamone); Matteo Zucca and Marcella Milillo for help with the data; Luigi Balletta, seminar participants in Livorno ( _Structural Change, Dynamics and Economic Growth_ ), Pisa, Petralia ( _VII Applied Economics Workshop_ ), and Naples ( _Old and New Forms of Organised and Serious Crime_ .) and four anonymous referees for comments. Gabriele Mellia, Alice Rizzuti and Giorgio Tortorici provided excellent research assistance. Financial support from MIUR (PRIN 2009, “Structural Change and Growth”), and University of Palermo (FFR 2012), is gratefully acknowledged. Usual caveat applies. 

## **Appendix A: Firm‑level variables** 

Table 3 contains the values of the correlation coefficients among the firm-level variables (standardized values) used in the econometric analysis of Sect. 5.2 ( _p_ values of tests for significance in parenthesis). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

21 

|Num. of empl.||||||1|
|---|---|---|---|---|---|---|
|Firm’s age|||||1|0 (0.93)|
|Debts/rev.||||1|− 0.01 (0.84)|0.02 (0.68)|
|Net profts|||1|− 0.04 (0.31)|− 0.18 (0)|0.03 (0.51)|
|Gross profts||1|0.36 (0)|0.04 (0.36)|0 (1)|0.13 (0)|
|Revenues|1|0.39 (0)|− 0.11 (0.01)|0 (0.98)|0.22 (0)|0.3 (0)|
|Pers. costs<br>1|0.72 (0)|0.47 (0)|− 0.07 (0.06)|0 (0.95)|0.17 (0)|0.26 (0)|
|Total assets<br>1<br>0.67 (0)|0.51 (0)|0.62 (0)|− 0.05 (0.25)|0.06 (0.17)|0.14 (0)|0.09 (0.02)|
|assets<br>nnel costs|nues|s profts|<br>rofts|<br>s/Rev.|’s age|. of empl.|
|Total<br>Perso|Reve|Gros|<br>Net p|<br>Debt|Firm|Num|



1 3 

European Journal of Law and Economics (2018) 46:1–37 

22 

## **Appendix B: On Palermo districts’ characteristics** 

In this appendix we discuss the distribution of the second-level variables in the Palermo districts. Figure 1 and Table 4 show the map of the districts. 

Figure 2 highlights the distribution of AP-firms across the districts. To take into account the spread of economic activity in the districts, we report the shares of APfirms over the number of limited-liability firms in each district.<sup>47</sup> Figure 2 shows that the spatial distribution of AP-firms is not homogeneous. The districts with the but highest shares of AP firms are located in the central-eastern part of the city,<sup>48</sup> 



**Fig. 1** Palermo districts: map 

47 We refer to the average number of limited-liability firms in a district in the period 2004–2012. 

48 The districts with the highest numbers of AP-firms are, starting from the East, Partanna-Mondello (9), Resuttana-San Lorenzo (17), Politeama (43), Liberta’ (25), and Malaspina-Palagonia (11). The western district of Brancaccio appears in the map similar to the former districts, but this is due to the very low number of AP-firms (2), combined with a very low number of registered firms in the district. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

23 

|**Table 4**Palermo districts:<br>names|ID|District name|
|---|---|---|
||1|Partanna Mondello|
||2|Tommaso Natale|
||3|Pallavicino|
||4|Monte Pellegrino|
||5|Arenella Vergine Maria|
||6|Resuttana San Lorenzo|
||7|Cruillas CEP|
||8|Borgo Nuovo|
||9|Uditore - Passo di Rigano|
||10|Malaspina-Palagonia|
||11|Liberta’|
||12|Politeama|
||13|Noce|
||14|Boccadifalco|
||15|Altarello|
||16|Zisa|
||17|Palazzo Reale - Monte di Pieta’|
||18|Tribunali-Castellammare|
||19|Cuba-Calatafmi|
||20|Mezzomonreale|
||21|Santa Rosalia|
||22|Oreto|
||23|Settecannoli|
||24|Brancaccio-Ciaculli|
||25|Villagrazia-Falsomiele|



there is a vast area including many peripheral districts in which no AP-firms are present.<sup>49</sup> 

Figures 3, 4 and 5 refer to measures of demographic structure, human capital and labour market conditions in the districts, while Tables 5, 6 and 7 present the descriptive statistics of all the second-level variables considered in this paper, while Table 8 shows their correlations. Figure 3, in particular, displays the population shares across the districts, that could represent a proxy for the potential market for a firm. Here no evidence of a clear spatial pattern appears: populous districts are present in both the central and peripheral areas of Palermo. 

Interestingly, a strongly divergent pattern is found when we observe human capital. Figure 4 shows that the spatial pattern of human capital, measured by the share of population with tertiary education, is similar to the one characterizing the presence of AP firms. This share can vary from approximately 20% in the some of the 

> 49 We refer in particular to the contiguous disticts of: Cruillas CEP, Borgo Nuovo, Boccadifalco, Mezzomonreale, Villagrazia-Falsomiele, Oreto and the district of Arenella - Vergine Maria. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

24 



**Fig. 2** Shares of AP firms in the 25 Palermo districts 

latter districts (e.g., Resuttana-San Lorenzo), to 2% in districts where no AP-firms are present (e.g. Arenella-Vergine Maria). In the latter districts a large majority of citizens has primary education only, approximately 60%, while in districts with higher presence of AP-firms this percentage drops to approximately 30% (similar proportions exist for elementary education levels).<sup>50</sup> These measures of human capital are strongly correlated with demographic indicators such as the shares of small and large families: high (low) human capital is correlated with low (high) family size, in line with the predictions of the child quality/quantity trade-off of the model of Becker et al. (1990). 

A similar pattern is found in Fig. 5. Unemployment rates are, respectively in the “high-AP” and “low-AP” districts, around 10 and 17%, a pattern reflected by the employment rates, which amount to, approximately, 85 and 70%. 

> 50 These averages are computed considering the groups of districts listed in Footnotes 48 and 49. The dependency ratios, i.e. the ratio of the share of population over 64 on the population under 14, are generally largely lower than 1 in the former and higher than 1 in the latter districts. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

25 



**Fig. 3** Shares of population in the 25 Palermo districts 

Tables 5, 6 and 7 report the statistics on the demographic characteristics of the districts, of the levels of human capital and on labour market indicators, while Tables 8 and 9 contain the correlations among these variables ( _p_ values of tests for significance in parenthesis). 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

26 



**Fig. 4** Shares of population with tertiary education in the 25 Palermo districts 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

27 



**Fig. 5** Unemployment rates in the 25 Palermo districts 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

28 

**Table 5** Demographic variables (districts) 

|Council|Tot. Pop.|Pop. share|Dep. Ratio|Family 123|Family 56|
|---|---|---|---|---|---|
|Palazzo Reale-Monte di Pieta’<br>I|11,352|0.02|0.69|0.71|0.12|
|Tribunali-Castellammare<br>I|10,137|0.01|0.86|0.73|0.12|
|Brancaccio-Ciaculli<br>II|15,618|0.02|0.45|0.54|0.18|
|Settecannoli<br>II|52,481|0.08|0.56|0.54|0.18|
|Oreto-Stazione<br>III|42,504|0.06|0.89|0.64|0.13|
|Villagrazia-Falsomiele<br>III|40915|0.06|0.62|0.56|0.17|
|Altarello<br>IV|16,944|0.02|0.53|0.57|0.15|
|Boccadifalco<br>IV|7909|0.01|0.47|0.59|0.16|
|Cuba<br>IV|23,587|0.03|0.91|0.63|0.12|
|Mezzomonreale<br>IV|38,567|0.06|0.7|0.58|0.14|
|Santa Rosalia<br>IV|25,678|0.04|1.03|0.64|0.13|
|Borgo Nuovo<br>V|21,085|0.03|0.74|0.56|0.2|
|Noce<br>V|29,940|0.04|0.95|0.67|0.11|
|Uditore-Passo di Rigano<br>V|33,331|0.05|0.88|0.63|0.12|
|Zisa<br>V|36,260|0.05|0.89|0.63|0.13|
|Cruillas-S. Giov. Ap. (ex C.E.P.)<br>VI|32998|0.05|0.54|0.55|0.15|
|Resuttana-San Lorenzo<br>VI|45,376|0.07|1.34|0.7|0.07|
|Arenella-Vergine Maria<br>VII|9299|0.01|0.64|0.59|0.15|
|Pallavicino<br>VII|27,428|0.04|0.48|0.55|0.19|
|Partanna Mondello<br>VII|16,652|0.02|0.77|0.68|0.09|
|Tommaso Natale<br>VII|21,125|0.03|0.54|0.59|0.14|
|Liberta’<br>VIII|45,002|0.07|1.6|0.75|0.06|
|Malaspina-Palagonia<br>VIII|21,793|0.03|1.88|0.74|0.06|
|Monte Pellegrino<br>VIII|29,011|0.04|0.86|0.65|0.12|
|Politeama<br>VIII|31,730|0.05|1.18|0.73|0.08|



Tot.Pop., total population; Pop. Share, population share; Dep. Ratio, dependency ratio; Family 123, share of families with 1–3 components; Family 56, share of families with 5–6 components or more 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

29 

**Table 6** Human capital (districts) 

|C|ouncil|Primm. Ed.|Prim. Ed.|Sec. Ed.|Tert. Ed.|Pop. Lit.|Pop. Illit.|
|---|---|---|---|---|---|---|---|
|Palazzo Reale-Monte di<br>Pieta’|I|0.32|0.61|0.11|0.05|0.17|0.06|
|Tribunali-Castellammare|I|0.27|0.54|0.16|0.1|0.16|0.05|
|Brancaccio-Ciaculli|II|0.31|0.65|0.15|0.02|0.15|0.03|
|Settecannoli|II|0.3|0.65|0.16|0.02|0.14|0.03|
|Oreto-Stazione|III|0.29|0.61|0.2|0.05|0.12|0.03|
|Villagrazia-Falsomiele|III|0.28|0.62|0.21|0.03|0.12|0.02|
|Altarello|IV|0.29|0.66|0.16|0.02|0.13|0.02|
|Boccadifalco|IV|0.23|0.58|0.21|0.06|0.12|0.02|
|Cuba|IV|0.24|0.55|0.26|0.07|0.1|0.02|
|Mezzomonreale|IV|0.22|0.56|0.28|0.06|0.09|0.01|
|Santa Rosalia|IV|0.28|0.61|0.2|0.05|0.11|0.03|
|Borgo Nuovo|V|0.31|0.67|0.13|0.02|0.13|0.04|
|Noce|V|0.27|0.59|0.21|0.06|0.11|0.02|
|Uditore-Passo di Rigano|V|0.21|0.51|0.29|0.1|0.09|0.01|
|Zisa|V|0.27|0.6|0.2|0.06|0.11|0.02|
|Cruillas-S. Giov. Ap. (ex<br>C.E.P.)|VI|0.24|0.57|0.25|0.04|0.11|0.03|
|Resuttana-San Lorenzo|VI|0.13|0.34|0.39|0.2|0.06|0.01|
|Arenella-Vergine Maria|VII|0.29|0.64|0.19|0.03|0.12|0.01|
|Pallavicino|VII|0.31|0.63|0.15|0.04|0.14|0.03|
|Partanna Mondello|VII|0.2|0.47|0.3|0.12|0.09|0.01|
|Tommaso Natale|VII|0.22|0.56|0.25|0.07|0.11|0.02|
|Liberta’|VIII|0.11|0.3|0.35|0.29|0.06|0|
|Malaspina-Palagonia|VIII|0.14|0.34|0.37|0.23|0.06|0|
|Monte Pellegrino|VIII|0.24|0.53|0.27|0.08|0.1|0.02|
|Politeama|VIII|0.18|0.41|0.26|0.21|0.1|0.02|



Primm. Ed.: share of population (age _>_ 6 ) with primary (elementary) education; Prim. Ed.: share of population (age _>_ 6 ) with primary (elementary and intermediate) education; Sec. Ed.: share of population (age _>_ 6 ) with secondary education; Tert. Ed.: share of population (age _>_ 6 ) with tertiary education; Pop. Lit.: share of population (age _>_ 6 ) with no education, literate; Pop. Illit.: share of population (age _>_ 6 ) with no education, illiterate 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

30 

|Empl. Indep.<br>0.16<br>0.23<br>0.15<br>0.16<br>0.15<br>0.17<br>0.15<br>0.17<br>0.16<br>0.17<br>0.14<br>0.13<br>0.17<br>0.18<br>0.17<br>0.16<br>0.22<br>0.18<br>0.17<br>0.29<br>0.22<br>0.25<br>0.22<br>0.18<br>0.26|. Constr., share|
|---|---|
|Empl. Dep.<br>0.83<br>0.76<br>0.83<br>0.83<br>0.83<br>0.82<br>0.84<br>0.81<br>0.83<br>0.82<br>0.85<br>0.85<br>0.82<br>0.81<br>0.82<br>0.83<br>0.78<br>0.8<br>0.82<br>0.7<br>0.77<br>0.75<br>0.77<br>0.81<br>0.73|nufacture; Empl|
|Empl. Serv.<br>0.82<br>0.81<br>0.71<br>0.76<br>0.79<br>0.78<br>0.75<br>0.78<br>0.83<br>0.83<br>0.83<br>0.75<br>0.81<br>0.84<br>0.83<br>0.79<br>0.86<br>0.74<br>0.77<br>0.82<br>0.75<br>0.89<br>0.9<br>0.8<br>0.87|of empl. in Ma|
|Empl. Constr.<br>0.07<br>0.05<br>0.07<br>0.06<br>0.06<br>0.06<br>0.08<br>0.09<br>0.04<br>0.05<br>0.05<br>0.08<br>0.06<br>0.05<br>0.06<br>0.07<br>0.04<br>0.06<br>0.07<br>0.06<br>0.07<br>0.03<br>0.03<br>0.05<br>0.04|pl. Manuf., share|
|Empl. Manuf<br>0.09<br>0.1<br>0.15<br>0.13<br>0.12<br>0.11<br>0.13<br>0.1<br>0.09<br>0.09<br>0.09<br>0.13<br>0.1<br>0.08<br>0.09<br>0.11<br>0.07<br>0.15<br>0.12<br>0.08<br>0.14<br>0.05<br>0.05<br>0.12<br>0.07|in Agriculture; Em<br>employed|
|Empl. Agr.<br>0.02<br>0.03<br>0.06<br>0.03<br>0.02<br>0.03<br>0.02<br>0.01<br>0.02<br>0.02<br>0.02<br>0.02<br>0.02<br>0.01<br>0.01<br>0.02<br>0.01<br>0.03<br>0.02<br>0.02<br>0.03<br>0.01<br>0.01<br>0.01<br>0.02|share of empl.<br>.: share of self-|
|Unempl.<br>0.27<br>0.19<br>0.2<br>0.2<br>0.19<br>0.17<br>0.2<br>0.16<br>0.15<br>0.14<br>0.18<br>0.21<br>0.17<br>0.14<br>0.17<br>0.18<br>0.08<br>0.16<br>0.21<br>0.14<br>0.17<br>0.07<br>0.07<br>0.15<br>0.1|mpl. Agr.,<br>Empl. Indep|
|Empl.<br>0.54<br>0.68<br>0.6<br>0.6<br>0.65<br>0.66<br>0.62<br>0.65<br>0.72<br>0.72<br>0.64<br>0.59<br>0.67<br>0.74<br>0.64<br>0.68<br>0.86<br>0.67<br>0.59<br>0.77<br>0.7<br>0.87<br>0.86<br>0.73<br>0.82|nt rate; E<br>ployees;|
|Council<br>I<br>I<br>II<br>II<br>III<br>III<br>IV<br>IV<br>IV<br>IV<br>IV<br>V<br>V<br>V<br>V<br>VI<br>VI<br>VII<br>VII<br>VII<br>VII<br>VIII<br>VIII<br>VIII<br>VIII|employme<br>share of em|
|Palazzo Reale-Monte di Pieta’<br>Tribunali-Castellammare<br>Brancaccio-Ciaculli<br>Settecannoli<br>Oreto-Stazione<br>Villagrazia-Falsomiele<br>Altarello<br>Boccadifalco<br>Cuba<br>Mezzomonreale<br>Santa Rosalia<br>Borgo Nuovo<br>Noce<br>Uditore-Passo di Rigano<br>Zisa<br>Cruillas-S. Giov. Ap. (ex C.E.P.)<br>Resuttana-San Lorenzo<br>Arenella-Vergine Maria<br>Pallavicino<br>Partanna Mondello<br>Tommaso Natale<br>Liberta’<br>Malaspina-Palagonia<br>Monte Pellegrino<br>Politeama|Empl., employment rate; Unempl.,<br>empl. in construction; Empl. Dep.,|



1 3 

European Journal of Law and Economics (2018) 46:1–37 

31 

|Pop. Illit.|||||1 (0)<br>− 0.8 (0)|0.88 (0)<br> 0.29 (0.155)|0.38 (0.057)|0.46 (0.021)|− 0.41<br>(0.041)|
|---|---|---|---|---|---|---|---|---|---|
|Pop. lit.||||1 (0)|0.89 (0)<br>− 0.91 (0)|0.93 (0)<br>0.54 (0.005)|0.66 (0)|0.68 (0)|− 0.71 (0)|
|Prim. Ed.||||1 (0)<br>0.89 (0)|0.75 (0)<br>− 0.96 (0)|0.92 (0)<br> 0.5 (0.01)|0.77 (0)|0.66 (0)|− 0.78 (0)|
|m. Ed.|||)|(0)<br>(0)|(0.001) <br>94 (0)|(0)<br>(0.023)|(0)|(0)|.84 (0)|
|Prim|||1 (0|0.96<br>0.81|0.62<br>− 0.|0.87<br>0.45|0.82|0.76|− 0|
|Ed.|||)<br>88 (0)|95 (0)<br>96 (0)|85 (0)<br>(0)|93 (0)<br>5 (0.01)|69 (0)|68 (0)|(0)|
|Sec.|||1 (0<br>− 0.|− 0.<br>− 0.|− 0.<br>0.95|− 0.<br>− 0.|− 0.|− 0.|0.73|
|Tert. Ed.||1 (0)|0.79 (0)<br>− 0.98 (0)|− 0.91 (0)<br>− 0.74 (0)|− 0.56<br>(0.004)<br>0.9 (0)|− 0.84 (0)<br> − 0.41<br>(0.043)|− 0.8 (0)|− 0.76 (0)|0.83 (0)|
||||||8)|8)||||
|. 56|)|.86 (0)|.77 (0)<br>8 (0)|2 (0)<br>3 (0)|2 (0.00<br>.84 (0)|4 (0)<br>2 (0.00|(0)|6 (0)|.86 (0)|
|Fam|1 (0|− 0|− 0<br>0.8|0.8<br>0.7|0.5<br> − 0|0.7<br>0.5|0.8|0.7|− 0|
|Fam. 123|1 (0)<br>− 0.89 (0)|0.8 (0)|0.48 (0.015) <br>− 0.76 (0)|− 0.61<br>(0.001)<br>− 0.42<br>(0.037)|− 0.15<br>(0.479)<br>0.62 (0.001)|− 0.5<br>(0.011)<br>− 0.43<br>(0.03)|− 0.76 (0)|− 0.7 (0)|0.81 (0)|
|Dep. ratio|1 (0)<br>0.8 (0)<br>− 0.82 (0)|0.86 (0)|0.7 (0)<br>− 0.84 (0)|− 0.74 (0)<br>− 0.73 (0)|− 0.47<br>(0.017)<br> 0.77 (0)|− 0.75 (0)<br>− 0.48<br>(0.016)|− 0.79 (0)|− 0.86 (0)|0.86 (0)|
|Pop. Sh.|1 (0)<br> 0.29 (0.167) <br>− 0.02<br>(0.917)<br>− 0.12<br>(0.56)|0.24 (0.25)|0.36 (0.077) <br>− 0.26<br>(0.207)|− 0.27<br>(0.195)<br>− 0.42<br>(0.035)|− 0.34<br>(0.097)<br> 0.29 (0.163)|− 0.34<br>(0.102)<br>− 0.13<br>(0.532)|− 0.25<br>(0.223)|− 0.46<br>(0.021)|0.32 (0.119)|
|tal Pop.|0)<br>0)<br>29 (0.167) <br>0.02<br>0.917)<br>0.12<br>0.56)|24 (0.25)|36 (0.077) <br>0.26<br>0.207)|0.27<br>0.195)<br>0.42<br>0.035)|0.34<br>0.097)<br>29 (0.163)|0.34<br>0.102)<br>0.13<br>0.532)|0.25<br>0.223)|0.46<br>0.021)|32 (0.119)|
|To|1 (<br>1 (<br>0.<br>−<br>(<br>−<br>(|0.|0.<br>−<br>(|−<br>(<br>−<br>(|−<br>(<br>0.|−<br>(<br>−<br>(|−<br>(|−<br>(|0.|
||al Pop.<br>. Sh.<br>p. ratio<br>m. 123<br>m. 56|t. Ed.|. Ed.<br>mm. Ed.|m. Ed.<br>. lit.|. Illit.<br>pl.|empl.<br>pl. Agr.|pl.<br>anuf.|pl.<br>onstr.|pl. Serv.|
||Tot<br>Pop<br>De<br>Fa<br>Fa|Ter|Sec<br>Pri|Pri<br>Pop|Pop<br>Em|Un<br>Em|Em<br>M|Em<br>C|Em|



1 3 

European Journal of Law and Economics (2018) 46:1–37 

32 

|Pop. lit.<br>Pop. Illit.|− 0.5<br>− 0.43|(0.012)<br>(0.031)<br>0.48 (0.016) 0.43 (0.033)<br>0.17 (0.415) 0.12 (0.563)|− 0.57<br>− 0.71 (0)|(0.003)|s; Family 56, share of families<br>n (age_>_6) with primary (ele-<br>e_>_6) with tertiary education;<br>l., employment rate; Unempl.,<br>in Construction; Empl. Dep.,<br>good/excellent conditions|
|---|---|---|---|---|---|
|Prim. Ed.|− 0.72 (0)|0.69 (0)<br>  0.07 (0.75)|− 0.44|(0.027)|3 component<br>re of populatio<br>opulation (ag<br>illiterate. Emp<br>hare of empl.<br>of buildings in|
|Primm. Ed.|− 0.76 (0)|0.73 (0)<br>0.06 (0.784)|− 0.28|(0.176)|milies with 1–<br>Prim. Ed., sha<br>Ed., share of p<br>no education,<br>mpl. Constr., s<br>. Good, share|
|. Ed.|(0.002)|.57<br>.003)<br>.17<br>.418)|(0.012)||are of fa<br>ucation;<br>n; Tert.<br>6) with<br>cture; E<br>water; H|
|Sec|0.6|− 0<br>(0<br>− 0<br>(0|0.5||3, sh<br>y) ed<br>catio<br>age_>_<br>anufa<br>ning|
||||)||12<br>r<br>du<br>(<br>M<br>un|
|Tert. Ed.|0.77 (0)|− 0.74 (0)<br>0 (0.998)|0.25 (0.227||ratio; Family<br>imary (elementa<br>th secondary e<br>re of population<br>re of empl. in<br>ouses with no r|
|Fam. 56|− 0.7 (0)|0.66 (0)<br> − 0.03<br>(0.892)|− 0.04|(0.852)|ependency<br>6) with pr<br>ge_>_6) wi<br>. Illit., sha<br>anuf., sha<br>share of h|
|Fam. 123|0.66 (0)|− 0.63<br>(0.001)<br>0.03 (0.876)|− 0.24|(0.246)|Dep. Ratio, d<br>ulation (age_>_ <br>population (a<br>n, literate; Pop<br>ture; Empl. M<br>; H. No Wat.,|
|Dep. ratio|0.47 (0.018)|− 0.43<br>(0.031)<br>− 0.19<br>(0.372)|0.08 (0.72)||pulation share;<br>d., share of pop<br>c. Ed., share of<br>ith no educatio<br>mpl. in Agricul<br>f self-employed|
|Pop. Sh.|− 0.03|(0.901)<br> 0.05 (0.803) <br>− 0.28<br>(0.18)|0.24 (0.253)||Pop. Share, po<br>ore. Primm. E<br>education; Se<br>n (age_>_6) w<br>gr., share of e<br>Indep., share o|
|Total Pop.|− 0.03|(0.901)<br>0.05 (0.803) <br>− 0.28<br>(0.18)|0.24 (0.253)||l population;<br>ponents or m<br>intermediate)<br>re of populatio<br>rate; Empl. A<br>oyees; Empl.|
||l. Dep.|l.<br>ep.<br>o Wat.|ood||op., tota<br>5–6 com<br>ary and<br>Lit., sha<br>oyment<br>of empl|
||Emp|Emp<br>Ind<br>H. N|H. G||Tot.P<br>with<br>ment<br>Pop.<br>empl<br>share|



1 3 

European Journal of Law and Economics (2018) 46:1–37 

33 

|Good|||||||0)|are of|build-|
|---|---|---|---|---|---|---|---|---|---|
|H.|||||||1 (|, sh|of|
|H. No Wat.||||||1 (0)|− 0.1 (0.65)|Empl. Constr.|H. Good, share|
|Indep.||||||4 (0.242)|5 (0.084)|ufacture;|ng water;|
|Empl.|||||1 (0)|− 0.2|− 0.3|in Man|o runni|
|l. Dep.|||||0)|(0.274)|(0.11)|f empl.|s with n|
|p||||0)|(|3|3|o|se|
|Em||||1 (|− 1|0.2|0.3|share|f hou|
|Serv.||||0.008)|6 (0.02)|2 (0.298)|0.869)|Manuf.,|, share o|
|l.|||)|(|4|2|(|l.|t.|
|Emp|||1 (0|0.51|− 0.|− 0.|0.03|Emp|o Wa|
|Constr.|||(0)|(0.028)|049)|.395)|.98)|iculture;|yed; H. N|
|.|||2|4|.|(0|(0|r|o|
|Empl||1 (0)|− 0.8|− 0.4|0.4 (0|0.18|0.01|. in Ag|f-empl|
|l. Manuf||(0)|97 (0)|53 (0.006)|(0.014)|(0.173)|09 (0.654)|re of empl|hare of sel|
|p|0)|1|0.|0.|8|8|0.|a|, s|
|Em|1 (|0.7|−|−|0.4|0.2|−|, sh|ep.|
|. Agr.|0)|0.161)|9 (0)|5 (0.47)|0.577)|0.532)|7 (0.725)|Empl. Agr.|Empl. Ind|
|Empl|1 (0)<br>0.66 (|0.29 (|− 0.6|− 0.1|0.12 (|0.13 (|− 0.0|rate;|loyees;|
|l.|047)<br>)|)|(0)|(0.001)|.001)|.487)|(0.007)|ployment|e of emp|
|mp|0.<br>(0|(0|1|3|(0|(0|2|m|ar|
|Une|1 (0)<br>0.4 (<br>0.67|0.73|− 0.7|− 0.6|0.61|0.15|− 0.5|l., e|p., sh|
||(0)<br>(0.043)<br>(0)|(0)|0)|0)|(0)|(0.754)|0.031)|te; Unemp|Empl. De|
|pl.|0)<br>0.97<br>0.41<br>0.71|0.76|6 (|3 (|0.71|0.07|3 (|t ra|on;|
|Em|1 (<br>−<br>−<br>−|−|0.7|0.7|−|−|0.4|men|ucti|
||.<br>pl.<br>. Agr.<br>. Manuf.|. Constr.|. Serv.|. Indep.|. Dep.|o Wat.|ood|., employ|in Constr|
||pl<br>em<br>pl<br>pl|pl|pl|pl|pl|N|G|pl|pl.|
||Em<br>Un<br>Em<br>Em|Em|Em|Em|Em|H.|H.|Em|em|



1 3 

European Journal of Law and Economics (2018) 46:1–37 

34 

## **Appendix C: On the stability of the coefficients** 

Table 10 presents the estimation of models aiming at proving the stability of the coefficients of the models of Table 2. In particular, we consider a model with only the two most significant firm-level variables from balance sheets (Model 1), then we add the two significant second-level variables (Model 2); firm’s age (Model 4), the balance-sheet variables that proved non-significant (Model 4) and finally the dummy variables (Model 5), corresponding to Model 5 of Table 2. 

**Table 10** Analysis of the stability of the coefficients of the models presented in Table 2 

||(1)|(2)|(3)|(4)|(5)|
|---|---|---|---|---|---|
|Personnel costs|1.519***|1.588***|1.684***|1.640***|1.012**|
||(0.396)|(0.402)|(0.424)|(0.465)|(0.441)|
|Capital stock|− 1.506***|− 1.578***|− 1.521***|− 1.671***|− 1.021**|
||(0.495)|(0.502)|(0.539)|(0.585)|(0.494)|
|Population share||− 0.380***|− 0.390***|− 0.371***|− 0.456***|
|||(0.129)|(0.132)|(0.137)|(0.153)|
|Elementary education share||− 0.226**|− 0.206*|− 0.187|− 0.239*|
|||(0.110)|(0.112)|(0.116)|(0.128)|
|Firm age|||− 0.0318***<br>(0.0107)|− 0.0272**<br>(0.0110)|− 0.0268**<br>(0.0117)|
|Revenues||||− 0.00963<br>(0.203)|− 0.201<br>(0.233)|
|Net profts<br>||||− 0.00528<br>(0.377)|− 0.0490<br>(0.394)|
|Gross profts||||0.378<br>(0.310)|0.345<br>(0.374)|
|Debt/revenues||||0.0382<br>(0.0300)|0.0353<br>(0.0294)|
|Employee Class I|||||0.998***<br>(0.304)|
|Employee Class II|||||1.769***<br>(0.369)|
|Construction dummy|||||− 1.183**<br>(0.519)|
|Services dummy|||||− 0.372<br>(0.411)|
|Constant|− 1.314***|− 1.272***|− 0.875***|− 0.962***|− 1.442***|
||(0.163)|(0.138)|(0.193)|(0.203)|(0.491)|
|N|633|633|622|587|558|



Standard errors in parentheses * _p_ < 0.10, ** _p_ < 0.05, *** _p_ < 0.010 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

35 

## **References** 

Agresti, A. (2002). _Categorical data analysis_ (2nd ed.). New York: Wiley. 

Alexander, B. (1997). The rational racketeer: Pasta protection in depression era Chicago. _The Journal of Law and Economics_ , _40_ , 175–202. 

Asmundo, A., & Lisciandra, M. (2008). The cost of protection racket in sicily. _Global Crime_ , _9_ , 221–240. Balletta, L., & Lavezzi, A. M. (2016). Extortion, firm’s size and the sectoral allocation of capital (in press). Battisti, M., Fioroni, T., Lavezzi, A. M., Masserini, L., & Pratesi, M. (2017a). The costs and benefits of resisting the extortion racket (in press). 

Battisti, M., Kourtellos, A., Durlauf, S., & Lavezzi, A. M. (2017b). Social interactions and crime prevention (in press). 

Becker, G. (1957). _The economics of discrimination_ . Chicago: University of Chicago Press. 

Becker, G. (1968). Crime and punishment: An economic approach. _The Journal of Political Economy_ , _76_ , 169–217. 

Becker, G., Murphy, K. M., & Tamura, R. (1990). Human capital, fertility and economic growth. _Journal of Political Economy_ , _98_ , S12–S37. 

Browne, W. J., & Draper, D. (2000). Implementation and performance issues in the Bayesian and likelihood fitting of multilevel models. _Computational Statistics_ , _15_ , 391–420. 

Bueno de Mesquita, E., & Hafer, C. (2007). Public protection or private extortion? _Economics and Politics_ , _20_ , 1–32. 

Cameron, A. C., & Trivedi, P. K. (2005). _Microeconometrics: Methods and applications_ . Cambridge: Cambridge University Press. 

Carlin, B. P., & Louis, T. A. (2000a). _Bayes and empirical bayes methods for data analysis_ (2nd ed.). Boca Raton: Chapman and Hall-CRC. 

Carlin, B. P., & Louis, T. A. (2000b). Empirical bayes: Past, present and future. _Journal of America Statistical Association_ , _95_ , 1286–1289. 

Clayton, D. G. (1996). Generalized linear mixed models. In W. R. Gilks, S. Richardson, & D. J. Spiegelhalter (Eds.), _Markov chain Monte Carlo in practice_ . London: Chapman and Hall. 

Confesercenti. (2010). XI Rapporto di Sos Impresa _Le mani della criminalita’ sulle imprese_ . Confesercenti. 

Congdon, P. (2006). _Bayesian models for categorical data_ . New York: Wiley. 

Demidenko, E. (2004). _Mixed models. Theory and applications_ . Hoboken, NJ: Wiley. 

Dixit, A. (2016). Anti-corruption institutions: Some history and theory, mimeo. In _International eco-_ 

_nomic association institutions, governance and corruption, conference montevideo (URY)_ . 

Draper, D. (2008). Bayesian multilevel analysis and MCMC. In J. de Leeuw (Ed.), _Handbook of quantitative multilevel analysis_ . New York: Springer. 

Efron, B., & Morris, C. (1973). Stein’s estimation rule and its competitors—An empirical Bayes approach. _Journal of the American Statistical Association_ , _68_ , 117–130. 

Efron, B., & Morris, C. (1975). Data analysis using Stein’s estimator and its generalizations. _Journal of the American Statistical Association_ , _70_ , 311–319. 

Fiasconaro, A. (2007). La citta’ avvolta dalla nube, _La Sicilia_ . Retrieved from http://www.addio pizzo .org/. 

Fioroni, T., Lavezzi, A. M., & Trovato, G. (2017). Organized crime, corruption and poverty traps (in press). Forno, F., & Gunnarson, C. (2010). Everyday shopping to fight the Mafia in Italy. In M. Micheletti & A. McFarland (Eds.), _Creative participation: Responsibility-taking in the political world_ . London: Paradigm Publisher. 

Gambetta, D. (1993). _The sicilian Mafia: The business of private protection_ . Cambridge: Harvard University Press. 

Gambetta, D. (2009). _Codes of the underworld_ . Princeton: Princeton University Press. Gambetta, D., & Reuter, P. (1995). Conspiracy among the many: The Mafia in legitimate industries. In G. Fiorentini & E. S. Peltzman (Eds.), _The economics of organised crime_ . Cambridge: Cambridge University Press. 

Gelfand, A., & Smith, A. (1990). Sampling-based approaches to calculating marginal densities. _Journal of the American Statistical Association_ , _85_ , 398–409. 

Goldstein, H. (2011). _Multilevel statistical models_ . New York: Wiley. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

36 

Gunnarson, C. (2014). Changing the game: Addiopizzo’s mobilisation against racketeering in Palermo. _European Review of Organised Crime_ , _1_ (1), 39–77. 

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). _Applied logistic regression_ (3rd ed.). New York: Wiley. 

Hox, J. J. (2010). _Multilevel analysis: Techniques and applications_ (2nd ed.). New York: Routledge. Keogh, R., & Cox, R. J. (2014). _Case-control studies_ . Cambridge: Cambridge University Press. Konrad, K. A., & Skaperdas, S. (1998). Extortion. _Economica_ , _65_ , 461–477. La Spina, A. (2008). Recent anti-Mafia strategies: The Italian experience. In D. Siegel & H. Nelen (Eds.), _Organized crime: Culture, markets and policies_ . New York: Springer. 

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. _Biometrics_ , _38_ , 963–974. 

Lavezzi, A. M. (2008). Economic structure and vulnerability to organised crime: Evidence from Sicily. _Global Crime_ , _9_ , 198–220. 

Lavezzi, A. M. (2014). Organised crime and the economy: A framework for policy prescriptions. _Global Crime_ , _15_ (1–2), 164–190. 

Maritz, J. S., & Lwin, T. (1989). _Empirical bayes methods_ . London: Chapman and Hall. 

Meridionews ( _Redazione_ ) (2015). Nuova intimidazione alla pasticceria Marsicano. ’Il rione mi boicotta e ora mi hanno levato le telecamere’. _Meridionews_ . Retrieved from http://merid ionew s.it/. 

Mete, V. (2011). I lavori di ammodernamento dell’autostrada Salerno-Reggio Calabria. Il ruolo delle grandi imprese nazionali. In R. Sciarrone (Ed.), _Alleanze nell’ombra_ . Rome: Donzelli. 

Morris, C. (1983). Parametric empirical bayes inference, theory and applications. _Journal of the American Statistical Association_ , _78_ , 47–65. 

Paoli, L. (2003). _Mafia brotherhoods. Organized crime, Italian style_ . Oxford: Oxford University Press. Partridge, H. (2012). The determinants of and barriers to critical consumption: A study of Addiopizzo. _Modern Italy_ , _17_ (3), 343–363. 

Pedrini, M., & Ferri, L. M. (2014). Socio-demographical antecedents of responsible consumerism propensity. _International Journal of Consumer Studies_ , _38_ (2), 127–138. 

Pinheiro, I. C., & Bates, D. M. (1995). Approximations to the log-likelihood function in nonlinear mixedeffects models. _Journal of Computational and Graphical Statistics_ , _4_ , 12–35. Pinotti, P. (2015). The economic costs of organized crime: Evidence from southern Italy. _Economic Journal_ , _125_ , F203–F232. 

Rabe-Hesketh, S., & Skrondal, A. (2006). Multilevel modelling of complex survey data. _Journal of the Royal Statistical Society, Series A_ , _169_ , 805–827. 

Rabe-Hesketh, S., Skrondal, A., & Pickles, A. (2004). Generalized multilevel structural equation modelling. _Psychometrika_ , _69_ , 167–190. 

Raudenbush, S. W., & Bryk, A. S. (2002). _Hierarchical linear models_ (2nd ed.). Thousand Oaks: Sage Publications. 

Schneider, J. C., & Schneider, P. T. (2003). _Reversible destiny. Mafia, antimafia and the struggle for Palermo_ . Oakland, CA: University of California Press. 

Skrondal, A., & Rabe-Hesketh, S. (2004). _Generalized latent variable modeling: Multilevel, longitudinal and structural equation models_ . Boca Raton, FL: Chapman & Hall/CRC. 

Skrondal, A., & Rabe-Hesketh, S. (2009). Prediction in multilevel generalised linear models. _Journal of the Royal Statistical Society, Series A_ , _172_ (3), 659–687. 

Snijders, T. A., & Bosker, R. (2011). _Multilevel analysis. An introduction to basic and advanced multilevel modelling_ (2nd ed.). Thousand Oaks, CA: Sage. 

Struttura e dimensione delle imprese Archivio Statistico delle Imprese Attive (ASIA). Anno 2005, Rome. Tuerlinckx, F., Rijmen, F., Verbeke, G., & De Boeck, P. (2006). Statistical inference in generalized linear mixed models: A review. _British Journal of Mathematical and Statistical Psychology_ , _59_ , 225–255. 

Vaccaro, A., & Palazzo, G. (2015). Values against violence: Institutional change in societies dominated by organized crime. _Academy of Management Journal_ , _58_ (4), 1075–1101. Vaccaro, A. (2012). To pay or not to pay? Dynamic transparency and the fight against the mafia’s extortionists. _Journal of Business Ethics_ , _106_ (1), 23–35. 

Van Dijk, J. (2007). Mafia markers: Assessing organized crime and its impact upon societies. _Trends in Organized Crime_ , _10_ , 39–56. 

Varese, F. (2014). Protection and extortion. In L. Paoli (Ed.), _Oxford handbook of organized crime_ (pp. 343–58). Oxford: Oxford University Press. 

Varese, F. (2009). The Camorra closely observed. _Global Crime_ , _10_ , 262–266. 

1 3 

European Journal of Law and Economics (2018) 46:1–37 

37 

Vroom, V. H., & Pahl, B. (1971). Relationship between age and risk taking among managers. _Journal of Applied Psychology_ , _55_ (5), 399. 

Ziniti, A. (2015). Fuga dalla Vucciria. Chiude il ristorante Santandrea. _La Repubblica_ . Retrieved from http://paler mo.repub blica .it/. 

1 3 


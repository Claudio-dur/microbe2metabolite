# E6 interpretation notes: the best-predicted metabolites and their literature support

## 1. What E6 found and how to read it

E6 explains 13 compounds — the 10 best predicted by ridge in PRISM cross-validation (one LC-MS feature
per HMDB compound) plus the best compound of three named classes (bile acids, indoles, straight-chain
fatty acids) — using permutation importance inside the same 5-fold PRISM CV over 155 samples (68 CD,
53 UC, 34 controls). No external cohort is touched (DESIGN.md decision #14). Ridge CV ρ runs from 0.834
(urobilin) down to 0.503 (butyrate/isobutyrate).

Importances are **small and noisy**. For ridge the drop in held-out ρ spans 0.0028–0.0363, and for 45 of
65 ridge (metabolite, taxon) pairs the fold-to-fold SD exceeds the mean (69%); for the MLP the range is
0.0038–0.0232 and SD exceeds mean for 15 of 65 pairs. Ridge and MLP top-5 lists agree on a median of 2
of 5 taxa (23 shared slots out of 65). Correlated species share the same signal and ridge spreads weight
across them. Signs are log-contrast coefficients on z-scored CLR columns ("+" = higher *relative*
abundance goes with more metabolite, given the rest of the composition), not fluxes. This is model
explanation, not causality (decision #29).

Compound names, classes and ρ values below are copied from `top_metabolites.csv`; taxa and signs from
`permutation_importance.csv` (ridge rows, importance ranks 1–3).

## 2. The 13 compounds

| Compound | Chemical class (as annotated) | ridge CV ρ | Top-3 taxa (ridge importance rank, coefficient sign) | Verdict |
|---|---|---|---|---|
| urobilin | Tetrapyrroles and derivatives | 0.834 | *Lachnospiraceae* sp. 7_1_58FAA (+), *Bacteroides vulgatus* (−), *Holdemania filiformis* (+) | class-level agreement, with caveats |
| eicosatrienoic acid | Long-chain fatty acids | 0.739 | *Eubacterium ramulus* (−), *Clostridium clostridioforme* (+), *Ruminococcus gnavus* (+) | co-variation, not production |
| docosapentaenoic acid | Very long-chain fatty acids | 0.726 | *Bacteroides uniformis* (−), *Butyricicoccus pullicaecorum* (+), *B. thetaiotaomicron* (−) | weakly consistent at best |
| sebacate | Fatty acids and conjugates | 0.720 | *Dorea* (unclassified) (−), *Clostridium leptum* (+), *Roseburia hominis* (+) | host/diet origin; microbial link indirect |
| ADMA | Arginine and derivatives | 0.719 | *Bacteroides cellulosilyticus* (−), *Leuconostoc citreum* (+), *Subdoligranulum* (unclassified) (−) | no microbial link found |
| undecanedionate | Medium-chain fatty acids | 0.718 | *Peptostreptococcaceae* (unclassified) (−), *Alistipes onderdonkii* (+), *Klebsiella pneumoniae* (−) | host/diet origin; microbial link indirect |
| 3-methyladipate-pimelate | Medium-chain fatty acids | 0.714 | *Burkholderiales* sp. 1_1_47 (+), *Parasutterella excrementihominis* (+), *B. pullicaecorum* (−) | plausible but unproven |
| linoleoyl ethanolamide | Organonitrogen compounds | 0.712 | *Ruminococcus gnavus* (+), *Alistipes senegalensis* (−), *Ruminococcus bromii* (−) | consistent with literature |
| pyridoxamine | Pyridoxamines | 0.709 | *Clostridium leptum* (+), *Anaerostipes hadrus* (+), *Bacteroides vulgatus* (−) | signs run against the capability map |
| pentadecanoate | Long-chain fatty acids | 0.705 | *Lachnospiraceae* sp. 8_1_57FAA (−), *Leuconostoc citreum* (−), *Roseburia intestinalis* (−) | diet biomarker; no gut-microbial route found |
| chenodeoxycholate | Bile acids, alcohols and derivatives | 0.695 | *Haemophilus parainfluenzae* (+), *Eubacterium* sp. 3_1_31 (−), *Rothia mucilaginosa* (+) | right chemistry, wrong taxa |
| tryptophan | Indoles and derivatives | 0.544 | *Bacteroides ovatus* (−), *Prevotella copri* (−), *Clostridium asparagiforme* (−) | consistent with literature |
| butyrate / isobutytare* | Straight chain fatty acids | 0.503 | *Clostridium bartlettii* (+), *Ruminococcus torques* (+), *Rothia mucilaginosa* (−) | partly consistent; isomers unresolved |

\* The compound name is reproduced verbatim from the annotation, typo included; the feature has no HMDB
ID and does not separate butyrate from isobutyrate. See §3.

## 3. Compound by compound

**Urobilin.** The clearest microbial case in the chemistry, though only partly in our taxa. Hall et al.
report that "Gut microorganisms are solely responsible for the reduction of bilirubin to urobilinogen",
identify the enzyme (BilR), find it "predominantly encoded by Firmicutes species", and state that "of the
tested species, all bilirubin reducers were found to be in the class Clostridia". They also report that
"no species within the Bacteroidaceae family were predicted to be bilirubin reducers", and that bilirubin
reductase was absent in only 0.1% of 1,801 healthy adult samples but absent in a significantly higher
fraction of Crohn's disease and ulcerative colitis samples (P < 2.2 × 10⁻¹⁶ for each, test of equal
proportions). In PRISM, Franzosa's largest co-varying cluster held 62 metabolite features, "all of them
elevated among controls", of which 11 were annotated as putative tetrapyrroles; Lloyd-Price list urobilin
among the features with the greatest differences between consecutive visits, "which showed larger
differences in individuals without IBD".

Two of our three signs line up with that picture: the top positive, *Lachnospiraceae* sp. 7_1_58FAA, is a
Clostridia member, and the negatives are *B. vulgatus* (Bacteroidaceae, the family Hall predicted to hold
no reducers) and *C. clostridioforme*, which Hall tested and did **not** find to be a reducer ("close
relatives of these reducers, such as *Clostridium clostridioforme* 2_1_49FAA and *Clostridium bolteae*
CC43_001B, were not found to be reducers") and which Franzosa list among species "uniquely DA and enriched
in CD". Two caveats pull the other way. *Holdemania filiformis*, our other strong positive, is in the class
**Erysipelotrichia**, not Clostridia (NCBI Taxonomy), so this is not a Clostridia-specific signal. And none
of the reducers or *bilR* carriers Hall names that are present in our 200-species table — *F. prausnitzii*,
*Roseburia intestinalis*, *R. inulinivorans*, *C. symbiosum*, *R. gnavus* — appears anywhere in the ridge
urobilin top 5; *R. gnavus* is the MLP's top urobilin taxon but carries a *negative* ridge coefficient
(−0.015, coefficient rank 66), the opposite of what its reducer status would suggest.

**Eicosatrienoic acid (ETA).** A host/dietary PUFA, not a known microbial product. Franzosa report that
"clusters 23 and 25 were elevated in CD and UC subjects and contained a variety of long-chain fatty acids,
including the standards arachidonic acid, adrenic acid, docosapentaenoic acid, and eicosatrienoic acid",
that "ETA and DPA are polyunsaturated long-chain fatty acids (PUFAs), and are examples of omega-3 and
omega-6 fatty acids, respectively", and that their "elevation in the IBD-afflicted gut may be explained by
higher rates of host cell death/turnover or reduced absorption from diet". They also note that PUFA
bactericidal activity "is particularly consistent with the negative correlations involving ETA (e.g. with
*Eubacterium ventriosum*)". Our signs are compatible with that pattern: *E. ramulus* — a flavonoid-degrading
gut commensal whose faecal population rises sharply after flavonoid intake (Simmering et al.) — negative,
and the CD-enriched *C. clostridioforme* and *R. gnavus* positive (*C. bolteae* is also positive at rank 5).
The most defensible reading is disease-linked co-variation, possibly with real growth effects on the
bacteria, not microbial synthesis of ETA.

**Docosapentaenoic acid (DPA).** Same host-membrane/dietary origin. Franzosa report that "conversely, DPA
associated positively with IBD-associated species, most notably *Ruminococcus gnavus*", and that among
their predicted positive associations "taurine and docosapentaenoic acid were confirmed to enhance growth"
in culture — an arrow from metabolite to microbe. Our DPA list does not reproduce that pair: it is led by
*Bacteroides uniformis* (−) and *B. thetaiotaomicron* (−) with *B. pullicaecorum* (+), and a positive sign
on *B. pullicaecorum*, a genus reduced in IBD stool (below), is hard to reconcile with DPA being
IBD-elevated. Weakly consistent at best.

**Sebacate and undecanedionate.** Medium-chain dicarboxylic acids (C10 and C11). In mammals, medium-chain
dicarboxylic acids "are the final products of microsomal ω-oxidation of long-chain fatty acids and the
subsequent peroxisomal β-oxidation of long-chain DCAs", and Houten et al. show that *Ehhadh*-null mice
"displayed a marked deficiency in the fasting-induced production of the medium-chain dicarboxylic acids
adipic and suberic acid". Note that Houten et al. study adipate (C6) and suberate (C8); neither sebacate
nor undecanedioate is examined there, and we found no established gut-bacterial route to either. Franzosa
flag dicarboxylic acids as a class whose IBD association was, "to the best of our knowledge, unique to this
study", so the covariance is real in PRISM but mechanistically unexplained.

One primary result does bear on the direction of any link: Fornelos et al. tested sebacic and dodecanedioic
acids on gut isolates and report that "*B. fragilis* growth was unaffected by sebacic and dodecanedioic
acids at single doses but strongly enhanced in the multiple-dose assay", while "*E. coli* growth was
moderately inhibited by sebacic and dodecanedioic acids" and no significant effect was seen on *R. gnavus*.
That is evidence a dicarboxylic acid can shape bacterial growth, not that bacteria make it. Our top taxa
mix health-associated Clostridia (*C. leptum* +, *R. hominis* + for sebacate) with IBD/facultative
organisms (*K. pneumoniae* − for undecanedionate); they are best read as disease-axis proxies.

**ADMA.** Asymmetric dimethylarginine is host-derived. Teerlink et al. state that "certain arginine
moieties on proteins are subject to asymmetric dimethylation by class 1 isoforms of protein arginine
methyltransferase (PRMT). After protein hydrolysis, ADMA is released within cells", and that humans
"generate approximately 300 μmol of ADMA per day, of which more than 80% is metabolized by DDAH"; Murphy
et al. describe the same two steps (PRMT methylation, then "the subsequent release from their respective
methylated protein(s) by proteolysis"). Neither review mentions gut bacteria, and Europe PMC searches for
ADMA together with gut microbiota or bacterial degradation returned no primary study of a bacterial route,
so we record **no known microbial link**. Of note, *H. parainfluenzae* is the fourth-ranked taxon (+) and
the MLP's top ADMA taxon; Schmidt et al. place *Haemophilus* among "core oral taxa" whose strains transmit
from mouth to gut and note that "increased presence of specific oral taxa in the intestine has in turn been
linked to several diseases, including … inflammatory bowel disease (IBD)". Its stool abundance plausibly
tracks that axis rather than ADMA chemistry.

**3-Methyladipate / pimelate.** Unresolved isomers, so two readings. Wierzbicki et al. studied
"ω-oxidation of PA to 3-methyl-adipic acid (3-MAA)" in adult Refsum disease and found 3-MAA excretion
correlated with plasma phytanic acid (r = 0.61; P = 0.03) and rose 208 ± 58% on fasting — i.e. 3-MAA tracks
a host ω-oxidation route acting on a dietary branched-chain fatty acid. (The further β-oxidation step
sometimes quoted is not stated in that abstract and we have not verified it.) Pimelate has a documented
microbial connection in the opposite direction, as a substrate: Magnúsdóttir et al. note that "biotin can be
synthesized *de novo* from two pimeloyl precursors, malonyl-ACP and pimelate" and examine the route "in
which BioW is used to convert salvaged pimelate to pimeloyl-CoA", with biotin biosynthesis "present in 40%
of the HGM genomes" — bacterial *consumption* of pimelate, not production. Our two strongest taxa are
*Burkholderiales* bacterium 1_1_47 (+), an unnamed relative, and *Parasutterella excrementihominis* (+),
which Ju et al. characterise as asaccharolytic (consistent with "the absence of the genes for transporting
and metabolizing exogenous sugars"), preferentially catabolising asparagine and aspartate, producing
"approximately 2.5 mM succinate after 72 h of growth", and whose colonisation reduced cholic and
taurocholic acid and altered tryptophan and tyrosine metabolites; its relative abundance "has been
associated with different host health outcomes such as inflammatory bowel disease, obesity, diabetes, and
fatty liver disease". Plausible but unproven, and no published pimelate link to either taxon was found.

**Linoleoyl ethanolamide (LEA).** The best-supported case, with the arrow pointing from metabolite to
microbe. Franzosa report that "cluster 13 (18 members) was uniquely elevated in CD subjects and enriched
for organonitrogen compounds, including the standards linoleoyl ethanolamide, palmitoylethanolamide, and
N-oleoylethanolamine". Fornelos et al. state that N-acylethanolamines are "produced primarily by the host",
that "PEA, OEA and LEA were enriched in CD samples only (FDR q<0.05), AEA was overabundant in both CD and
UC", and that "NAEs promote the growth of species overabundant in IBD and restrict the growth of those
depleted in IBD", with *Alistipes* and *Bacteroides* species decreasing "upon treatment with all NAEs except
PEA". For LEA specifically they report that "LEA inhibited the growth of *B. cellulosilyticus* and enhanced
that of *E. coli*, *R. gnavus* and *B. producta*". Our signs track that direction: *R. gnavus* (+), and
*Alistipes senegalensis* (−), *R. bromii* (−), *R. hominis* (−) at ranks 2, 3 and 5.

*(Correction from an earlier draft: that draft stated LEA inhibits* B. fragilis *while* R. gnavus *and*
E. coli *are largely unaffected. Both the species and the direction were wrong — the paper reports
inhibition of* B. cellulosilyticus *and enhancement of* E. coli*,* R. gnavus *and* B. producta*. The
corrected finding fits our positive sign on* R. gnavus *more directly than the draft's version did.)*

**Pyridoxamine (vitamin B6).** Gut microbes do synthesise B6: Magnúsdóttir et al. find "the *de novo*
biosynthesis of pyridoxal 5′-phosphate is present in 50% of the tested HGM genomes", that "only genomes
from the Bacteroidetes and Proteobacteria phyla contained the roles PdxJ and PdxH", and that "only few
Firmicutes and Fusobacteria can synthesize pyridoxal 5′-phosphate, but those that do use the shorter PdxTS
route". Two caveats: that paper addresses pyridoxal 5′-phosphate and pyridoxine, never pyridoxamine
specifically; and our directions run opposite to its capability map — Firmicutes positive (*C. leptum*,
*A. hadrus*, and *R. torques* at rank 4) with the Bacteroidetes member *B. vulgatus* negative — so the
signal is unlikely to be reading biosynthetic capacity. B vitamins are broadly depleted in IBD stool
(Franzosa: "pantothenate (down in IBD)"; Lloyd-Price: "pantothenate and nicotinate (vitamins B5 and B3,
respectively) were particularly depleted in the gut during IBD"), but neither paper reports pyridoxamine
individually — we searched both full texts for "pyridox" and found no hit.

**Pentadecanoate (C15:0).** An odd-chain saturated fatty acid used as a dairy-fat intake biomarker; Jenkins
et al. note the interest in "the possible application of C15:0 in blood as a marker for intake of milk fat"
and explain that odd-chain saturated fatty acids "are produced in relatively high levels by rumen microbial
fermentation and microbial de-novo lipogenesis which then transfers into the host animal". The propionate
route they describe is for **C17:0 in rumen microbes** ("in certain microbes C17:0 can be produced where
propionic acid … is trapped by the rumen bacteria/protozoa"), not for C15:0 in humans, and that review says
nothing about the human colonic microbiota producing these fatty acids. Franzosa's class-level test puts
long-chain fatty acids among the most numerous depleted classes in IBD ("triterpenoids and long-chain fatty
acids (LCFAs) … were the most numerous depleted classes (total n=135 and 111)") while individual LCFAs such
as ETA and DPA sat in CD-elevated clusters, so the class direction there is mixed. Read as diet-driven; the
mostly negative Firmicutes signs are co-variation.

**Chenodeoxycholate (CDCA).** The compound's biology is microbial, but our taxa are not the known actors.
CDCA is a host-made primary bile acid. Jones et al. describe deconjugation as the gateway step — "the
initial 'gateway' reaction in the bacterial metabolism of CBAs is mediated by bile salt hydrolase (BSH) …
which catalyzes the deconjugation of CBAs to liberate free primary bile acids (BAs; cholic acid or
chenodeoxycholic acid) and amino acids" — and show it is widely redundant: "functional BSHs are distributed
among two domains of life and all major bacterial divisions of the human gut microbiota" (Firmicutes 30%,
Bacteroidetes 14.4%, Actinobacteria 8.9%, plus two archaea), with the "wide phylogenetic distribution and
high level of redundancy … indicative of selection for this activity at the host level". The
7α-dehydroxylation that would turn CDCA into lithocholate is the opposite: Ridlon et al. state that "bile
acid 7-dehydroxylation appears restricted to a limited number of intestinal anaerobes representing a small
fraction of the total colonic flora", and Funabashi et al., working in *C. scindens* with orthologues from
*C. hylemonae* and *C. hiranonis*, put such organisms at "an estimated 1:10⁶ in a typical gut community"
and describe a *bai* operon of "eight genes: seven encode enzymes and the eighth, *baiG*, encodes a
transporter".

The IBD pattern is textbook: Franzosa found "IBD-enriched bile acids included cholate (q=0.003) and
chenodeoxycholate (q=0.0002)" with non-significant complementary depletions of lithocholate and
deoxycholate (q = 0.06 and 0.13), and Lloyd-Price found "the secondary bile acids lithocholate and
deoxycholate (q = 5 × 10⁻⁷ and q = 1.8 × 10⁻⁴, respectively) were reduced in dysbiosis, suggesting that
secondary bile-acid producing bacteria are depleted in IBD-related dysbiosis, or that transit time through
the colon is too short for these compounds to be metabolized". But our top taxa are *H. parainfluenzae* (+),
*R. mucilaginosa* (+) and *B. vulgatus* (+, rank 5) — markers of the IBD/oral-taxon axis rather than
bile-acid chemistry. *C. scindens* is the only one of the three *bai*-carrying species in our 200-species
table (*C. hylemonae* and *C. hiranonis* are absent from it entirely); it clears the 10% prevalence floor
at 17.4% of PRISM samples yet appears nowhere in the importance table for either model. At that prevalence
and abundance it carries little CLR variance — exactly the dilution the caveat predicts.

**Tryptophan.** Consistent. Tryptophan is a dietary essential amino acid that gut bacteria consume. Roager
& Licht state that "indole formation occurs via the action of the enzyme tryptophanase (TnaA), which is
expressed in many Gram-negative, as well as Gram-positive bacterial species including *Escherichia coli,
Clostridium* spp. and *Bacteroides* spp.", and their Table 1 lists *Bacteroides ovatus* under indole,
indoleacetic acid and indolelactic acid. They also note that "*Clostridium sporogenes* converts tryptophan
into tryptamine, indolelactic acid (ILA) and indolepropionic acid (IPA)" and that *Peptostreptococcus* spp.
convert tryptophan to indoleacrylic acid and IPA "possibly due to the presence of the phenyllactate
dehydratase gene cluster (*fldAIBC*)"; Dodd et al. characterised that *C. sporogenes* aromatic-amino-acid
pathway, with tryptophan among its three substrates. In a recent in-vitro survey of eight commensals,
Tingler et al. report that these bacteria "differentially consumed glutamine, glutamate, and tryptophan"
and that "*P. copri* was the sole producer of tryptamine". Our two strongest taxa, *B. ovatus* and
*P. copri*, both carry negative coefficients — more of these consumers, less residual stool tryptophan —
the expected direction. Franzosa also report "a decrease in tryptophan metabolism genes in microbiome
samples from patients with Crohn's disease". (*P. copri* is not mentioned in Roager & Licht; that species
rests on Tingler et al. alone.)

**Butyrate / isobutyrate.** Partly consistent, and the annotation explains why: the feature does not
separate butyrate (fibre fermentation) from isobutyrate. Rios-Covian et al. state that "it is known that
BCFA are mainly produced during fermentation of branched chain amino acids (valine, leucine, and isoleucine)
by the intestinal microbiota", that "in the human intestine, the fermentation of branched chain amino acids
is carried out mainly by genera *Bacteroides* and *Clostridium*", and that "BCFA have been proposed as
markers of colonic protein fermentation". The canonical butyrate producers are all in the input table at
high prevalence — Louis & Flint describe "the two most abundant groups related to *Eubacterium
rectale/Roseburia* spp. and to *Faecalibacterium prausnitzii*" (*F. prausnitzii* 88.4% and *E. rectale*
89.0% prevalence here), Duncan et al. identified lactate-utilising butyrate producers related to
*Eubacterium hallii* and *Anaerostipes caccae* while "significant ability to utilize lactate was not
detected in … *Roseburia intestinalis*, *Eubacterium rectale*, or *Faecalibacterium prausnitzii*", and
Eeckhaut et al. found genus-level *Butyricicoccus* counts "significantly lower compared with healthy
subjects (9.32 log10/g stool)" in active and remission UC and CD — yet none of them leads our list
(*F. prausnitzii*, *E. rectale*, *C. symbiosum* and *E. hallii* appear nowhere in the importance table).

Instead the top taxa are an amino-acid fermenter, *C. bartlettii* (+), which Roager & Licht's Table 1 lists
under skatole, indoleacetic acid and indolelactic acid, and a mucin degrader, *R. torques* (+), which Png
et al. found "~100-fold" increased (with *R. gnavus* ">4-fold") "in macroscopically and histologically
normal intestinal epithelium of both CD and UC". *O. splanchnicus* carries a negative sign at rank 5, which
is hard to read: Vital et al. report that "groups linked to *Odoribacter splanchnicus* and *Alistipes
putredinis* … dominate the lysine pathway", but they never characterise *O. splanchnicus* as a confirmed
butyrate producer, and in their survey "the acetyl-CoA pathway was dominating for almost all individuals
(mean, 79.7%)", so the lysine route is a minor contributor. Lloyd-Price found "SCFAs were generally reduced
in dysbiosis", with "the reduction in butyrate in particular … consistent with the previously observed
depletion of butyrate producers such as *F. prausnitzii* and *R. hominis*"; in Franzosa, "while not
statistically significant in this cohort, the SCFAs butyrate and propionate were decreased in both UC and
CD relative to controls".

## 4. What this does and does not show

Prediction here rests on **co-variation within one cross-sectional cohort of 155 people**, not on metabolic
flux. Disease status, mucosal inflammation, diet, medication and transit time plausibly move microbes and
metabolites together — Lloyd-Price note that "metabolite pools were less diverse in individuals with IBD …
this might be caused by poor nutrient absorption, greater water or blood content in the bowels, and shorter
bowel transit times" — so a species can be a good predictor purely as a disease marker. That is the most
parsimonious reading for the fatty acids, the dicarboxylic acids, ADMA and pentadecanoate, and for the
oral/upper-GI commensals (*H. parainfluenzae*, *R. mucilaginosa*) that surface for chemically unrelated
compounds. Franzosa make the same general point: "gut metabolite profiles are jointly derived from diet,
modified human metabolites and microbially-derived compounds", and clusters of co-varying metabolites can
arise from "metabolites co-contributed from a specific dietary source" as readily as from "metabolites
co-produced by a specific microbe".

Single-species importance is **diluted across correlated species**: ridge spreads weight over co-abundant
Clostridia, so each permutation drop is small (ridge maximum 0.036 ρ) and fold-unstable (SD > mean for 69%
of ridge pairs), and a genuine but rare catalyst such as *C. scindens* can be the right mechanism and still
rank nowhere. Permuting a CLR column also breaks the compositional constraint, so the perturbed input is
off-manifold and the importance is a property of this model, not of the microbiome. Signs are log-contrast
coefficients: "+" means higher relative abundance of that taxon *given the rest of the composition*.

Nothing here establishes direction of causation, and where the literature does speak to direction it often
points the other way: for LEA and for DPA the cited experiments have the metabolite affecting bacterial
growth, not bacteria producing the metabolite. Treat the whole table as hypothesis generation — strongest
where an independent mechanism exists (LEA, tryptophan, and the chemistry, though not the taxa, of urobilin
and chenodeoxycholate) and weakest where only the disease axis is shared.

## 5. Claims we could not verify

- **ADMA and gut bacteria.** No primary evidence was found that gut bacteria produce, consume or degrade
  ADMA. Europe PMC searches for ADMA with gut microbiota, and for bacterial ADMA degradation, returned no
  relevant primary study. The host biochemistry above is verified from two reviews we read (Teerlink 2009;
  Murphy 2016), neither of which mentions gut bacteria.
- **IBD direction for five compounds.** Neither Franzosa 2019 nor Lloyd-Price 2019 reports pyridoxamine,
  sebacate, undecanedioate, 3-methyladipate/pimelate or pentadecanoate as individually differentially
  abundant. We searched both full texts for "pyridox", "sebac", "undecanedio", "pimel", "methyladip",
  "pentadecan" and "dicarboxyl" and found no hits (Franzosa discusses dicarboxylic acids only as a class).
  The IBD direction of change for these five compounds in PRISM is **not** established here.
- ***Rothia mucilaginosa*.** No published report of *R. mucilaginosa* enrichment in IBD stool was verified,
  and it is not named in Schmidt et al. 2019 (no "Rothia" in the text). Its description as an oral-cavity
  commensal is background knowledge, not a cited finding; the Schmidt citation covers *Haemophilus* only.
- **Taxa with no verified function.** *Leuconostoc citreum*, *Megasphaera* (unclassified), *Burkholderiales*
  bacterium 1_1_47, *Holdemania filiformis*, *Clostridium asparagiforme*, *Alistipes senegalensis*,
  *A. onderdonkii*, *Dorea* (unclassified), *Subdoligranulum* (unclassified) and *Oscillibacter*
  (unclassified): no compound-specific metabolic function was verified for any of these. They appear above
  only as data. (*H. filiformis*'s class assignment is from NCBI Taxonomy, not from a functional study.)
- **Compound identity.** HMDB0002925 could not be read directly (hmdb.ca returned HTTP 403), so the exact
  ETA isomer behind the LC-MS feature is unconfirmed; we rely on Franzosa's own statement that ETA is an
  omega-3 example. More generally, whether each selected LC-MS feature is chemically the compound named was
  not independently checked — the annotation is "putative" for chemical class, and Franzosa note that
  starred standards have isomeric forms that could not be differentiated. The butyrate/isobutyrate feature
  has no HMDB ID, so which isomer drives the signal cannot be determined from these data.
- **Recency.** Tingler et al. was verified only through its Europe PMC record and abstract (PMC12860712,
  PMID 41630915); we did not read the full text or find independent replication of the *P. copri* tryptamine
  result, which comes from an eight-species in-vitro panel. Its year is given as 2025 by the PMC record and
  the DOI, although the issue is listed as volume 29(2).
- **Source access.** Ridlon et al. 2006 has no PMC full text; only its abstract was read, and the *bai*
  operon gene count above is cited from Funabashi et al. instead. Franzosa, Lloyd-Price, Hall, Fornelos,
  Magnúsdóttir, Roager & Licht, Jenkins and Jones were read as converted page text, in some cases only the
  first ~100,000 characters plus targeted re-reads, so an "absent" result is strong but not exhaustive.
- **Prevalence figures.** Taxon prevalences quoted here were recomputed over all 155 PRISM samples after
  re-closing each sample over its 200 taxa, matching `prevalent_taxa` in `src/m2m/preprocess.py`
  (abundance_threshold 1.0e-4, min_prevalence 0.10; 151 of 200 taxa pass). The pipeline refits this filter
  per training fold, so the exact kept set varies slightly by fold. (An earlier draft quoted *F. prausnitzii*
  87.1% and *E. rectale* 88.4%, computed on the un-reclosed table; the correct values under the pipeline's
  rule are 88.4% and 89.0%.)
- **Confounding not tested.** The claim that transit time, medication and diet move microbes and metabolites
  together is supported in general terms by Franzosa and Lloyd-Price but was not tested on PRISM in E6; no
  covariate-adjusted analysis was run.

## Sources

All DOIs and identifiers below were resolved and the cited statements read, except where §5 says otherwise.

- Franzosa et al. 2019, *Nat Microbiol* (PRISM cohort) — [10.1038/s41564-018-0306-4](https://doi.org/10.1038/s41564-018-0306-4) ([PMC6342642](https://pmc.ncbi.nlm.nih.gov/articles/PMC6342642/))
- Lloyd-Price et al. 2019, *Nature* — [10.1038/s41586-019-1237-9](https://doi.org/10.1038/s41586-019-1237-9) ([PMC6650278](https://pmc.ncbi.nlm.nih.gov/articles/PMC6650278/))
- Hall et al. 2024, *Nat Microbiol*, "BilR is a gut microbial enzyme that reduces bilirubin to urobilinogen" — [10.1038/s41564-023-01549-x](https://doi.org/10.1038/s41564-023-01549-x) ([PMC10769871](https://pmc.ncbi.nlm.nih.gov/articles/PMC10769871/))
- Fornelos et al. 2020, *Nat Microbiol*, "Growth effects of N-acylethanolamines on gut bacteria reflect altered bacterial abundances in inflammatory bowel disease" — [10.1038/s41564-019-0655-7](https://doi.org/10.1038/s41564-019-0655-7) ([PMC7047597](https://pmc.ncbi.nlm.nih.gov/articles/PMC7047597/))
- Funabashi et al. 2020, *Nature*, "A metabolic pathway for bile acid dehydroxylation by the gut microbiome" — [10.1038/s41586-020-2396-4](https://doi.org/10.1038/s41586-020-2396-4) ([PMC7319900](https://pmc.ncbi.nlm.nih.gov/articles/PMC7319900/))
- Ridlon, Kang & Hylemon 2006, *J Lipid Res*, "Bile salt biotransformations by human intestinal bacteria" — [10.1194/jlr.R500013-JLR200](https://doi.org/10.1194/jlr.R500013-JLR200) (PMID 16299351; abstract only)
- Jones et al. 2008, *PNAS* (BSH) — [10.1073/pnas.0804437105](https://doi.org/10.1073/pnas.0804437105) ([PMC2533232](https://pmc.ncbi.nlm.nih.gov/articles/PMC2533232/))
- Roager & Licht 2018, *Nat Commun*, "Microbial tryptophan catabolites in health and disease" — [10.1038/s41467-018-05470-4](https://doi.org/10.1038/s41467-018-05470-4) ([PMC6098093](https://pmc.ncbi.nlm.nih.gov/articles/PMC6098093/))
- Dodd et al. 2017, *Nature*, "A gut bacterial pathway metabolizes aromatic amino acids into nine circulating metabolites" — [10.1038/nature24661](https://doi.org/10.1038/nature24661) ([PMC5850949](https://pmc.ncbi.nlm.nih.gov/articles/PMC5850949/), PMID 29168502)
- Tingler et al. 2025, *iScience*, "Commensal human gut microbes produce species specific neuroactive compounds" — [10.1016/j.isci.2025.114424](https://doi.org/10.1016/j.isci.2025.114424) ([PMC12860712](https://pmc.ncbi.nlm.nih.gov/articles/PMC12860712/), PMID 41630915; abstract only)
- Louis & Flint 2009, *FEMS Microbiol Lett* — [10.1111/j.1574-6968.2009.01514.x](https://doi.org/10.1111/j.1574-6968.2009.01514.x) (PMID 19222573; abstract only)
- Duncan, Louis & Flint 2004, *Appl Environ Microbiol* — [10.1128/AEM.70.10.5810-5817.2004](https://doi.org/10.1128/AEM.70.10.5810-5817.2004) ([PMC522113](https://pmc.ncbi.nlm.nih.gov/articles/PMC522113/))
- Vital, Howe & Tiedje 2014, *mBio* — [10.1128/mBio.00889-14](https://doi.org/10.1128/mBio.00889-14) ([PMC3994512](https://pmc.ncbi.nlm.nih.gov/articles/PMC3994512/))
- Eeckhaut et al. 2013, *Gut*, "*Butyricicoccus pullicaecorum* in inflammatory bowel disease" — [10.1136/gutjnl-2012-303611](https://doi.org/10.1136/gutjnl-2012-303611) (PMID 23263527; abstract only)
- Rios-Covian et al. 2020, *Front Microbiol* (branched SCFA) — [10.3389/fmicb.2020.00973](https://doi.org/10.3389/fmicb.2020.00973) ([PMC7271748](https://pmc.ncbi.nlm.nih.gov/articles/PMC7271748/))
- Png et al. 2010, *Am J Gastroenterol* (mucolytic bacteria) — [10.1038/ajg.2010.281](https://doi.org/10.1038/ajg.2010.281) (PMID 20648002; abstract only)
- Magnúsdóttir et al. 2015, *Front Genet* (B vitamins) — [10.3389/fgene.2015.00148](https://doi.org/10.3389/fgene.2015.00148) ([PMC4403557](https://pmc.ncbi.nlm.nih.gov/articles/PMC4403557/))
- Jenkins, West & Koulman 2015, *Molecules* (odd-chain fatty acids) — [10.3390/molecules20022425](https://doi.org/10.3390/molecules20022425) ([PMC6272531](https://pmc.ncbi.nlm.nih.gov/articles/PMC6272531/))
- Houten et al. 2012, *J Lipid Res* (medium-chain dicarboxylic acids) — [10.1194/jlr.M024463](https://doi.org/10.1194/jlr.M024463) ([PMC3371241](https://pmc.ncbi.nlm.nih.gov/articles/PMC3371241/))
- Ju et al. 2019, *ISME J* (*Parasutterella*) — [10.1038/s41396-019-0364-5](https://doi.org/10.1038/s41396-019-0364-5) ([PMC6776049](https://pmc.ncbi.nlm.nih.gov/articles/PMC6776049/))
- Schmidt et al. 2019, *eLife* (oral–gut transmission) — [10.7554/eLife.42693](https://doi.org/10.7554/eLife.42693) ([PMC6424576](https://pmc.ncbi.nlm.nih.gov/articles/PMC6424576/))
- Simmering, Pforte, Jacobasch & Blaut 2002, *FEMS Microbiol Ecol* (*E. ramulus*) — [10.1111/j.1574-6941.2002.tb00957.x](https://doi.org/10.1111/j.1574-6941.2002.tb00957.x) (PMID 19709232; abstract only)
- Wierzbicki et al. 2003, *J Lipid Res* (phytanic ω-oxidation, 3-methyladipate) — [10.1194/jlr.M300121-JLR200](https://doi.org/10.1194/jlr.M300121-JLR200) (PMID 12700346; abstract only)
- Kabeerdoss et al. 2013, *BMC Gastroenterol* (*C. leptum* group in IBD) — [10.1186/1471-230X-13-20](https://doi.org/10.1186/1471-230X-13-20) ([PMC3565871](https://pmc.ncbi.nlm.nih.gov/articles/PMC3565871/))
- Teerlink, Luo, Palm & Wilcox 2009, *Pharmacol Res*, "Cellular ADMA: regulation and action" — [10.1016/j.phrs.2009.08.002](https://doi.org/10.1016/j.phrs.2009.08.002) ([PMC2767414](https://pmc.ncbi.nlm.nih.gov/articles/PMC2767414/))
- Murphy, Tommasi, Lewis & Mangoni 2016, *Molecules* (DDAH inhibitors) — [10.3390/molecules21050615](https://doi.org/10.3390/molecules21050615) ([PMC6273216](https://pmc.ncbi.nlm.nih.gov/articles/PMC6273216/))
- NCBI Taxonomy, *Holdemania filiformis* (class Erysipelotrichia) — [taxonomy browser](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?name=Holdemania+filiformis)

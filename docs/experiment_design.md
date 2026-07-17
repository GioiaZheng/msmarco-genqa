# Experiment Design: Metric and Paired-Inference Definitions

This document is the mathematical contract for the repository's evaluation
layer. The equations below describe the current implementations; they do not
promote any single metric to a complete measure of RAG quality.

## 1. Notation

For a stage-specific evaluated query set

$$
\mathcal Q=\{q_1,\ldots,q_N\},
\qquad N=|\mathcal Q|,
$$

let

$$
\begin{aligned}
L_q &= (d_{q,1},d_{q,2},\ldots),
&R_q&=\{d:\operatorname{rel}_q(d)>0\},\\
\hat y_q&=\text{generated prediction},
&\mathcal A_q&=\{a_{q,1},\ldots,a_{q,J_q}\},\\
\mathcal C_q&=(c_{q,1},\ldots,c_{q,K}),
&\mathbf 1[\cdot]&=\text{indicator function}.
\end{aligned}
$$

For any per-query metric $m_q$,

$$
\overline m
=
\frac{1}{N}\sum_{q\in\mathcal Q}m_q.
$$

## 2. Retrieval Metrics

The retrieval macro-average includes only queries with at least one positive
qrel:

$$
\mathcal Q_{\mathrm{ret}}^{+}
=
\{q\in\mathcal Q:R_q\neq\varnothing\},
\qquad
N_{\mathrm{ret}}=|\mathcal Q_{\mathrm{ret}}^{+}|.
$$

### 2.1 Reciprocal Rank and MRR@$k$

$$
r_q^{(k)}
=
\min\{i\le k:d_{q,i}\in R_q\},
$$

$$
\operatorname{RR}_q@k
=
\begin{cases}
\dfrac{1}{r_q^{(k)}}, & r_q^{(k)}\text{ exists},\\[6pt]
0, & \text{otherwise},
\end{cases}
$$

$$
\operatorname{MRR}@k
=
\frac{1}{N_{\mathrm{ret}}}
\sum_{q\in\mathcal Q_{\mathrm{ret}}^{+}}\operatorname{RR}_q@k.
$$

### 2.2 Recall@$k$

$$
\operatorname{Recall}_q@k
=
\frac{
\left|\{d_{q,1},\ldots,d_{q,k}\}\cap R_q\right|
}{|R_q|},
$$

$$
\operatorname{Recall}@k
=
\frac{1}{N_{\mathrm{ret}}}
\sum_{q\in\mathcal Q_{\mathrm{ret}}^{+}}\operatorname{Recall}_q@k.
$$

### 2.3 Binary nDCG@$k$

The current MS MARCO `dev/small` path uses binary gain

$$
g_q(d)=\mathbf 1[d\in R_q].
$$

Hence

$$
\operatorname{DCG}_q@k
=
\sum_{i=1}^{k}
\frac{g_q(d_{q,i})}{\log_2(i+1)},
$$

$$
\operatorname{IDCG}_q@k
=
\sum_{i=1}^{\min(k,|R_q|)}\frac{1}{\log_2(i+1)},
$$

$$
\operatorname{nDCG}_q@k
=
\frac{\operatorname{DCG}_q@k}{\operatorname{IDCG}_q@k},
\qquad
\operatorname{nDCG}@k
=
\frac{1}{N_{\mathrm{ret}}}
\sum_{q\in\mathcal Q_{\mathrm{ret}}^{+}}\operatorname{nDCG}_q@k.
$$

## 3. Reference-Based Generation Metrics

Let $\nu(\cdot)$ denote the repository's lowercase, punctuation/article
removal, and whitespace normalization, and let $T_{\nu}(x)$ be the resulting
token multiset.

### 3.1 Exact Match

$$
\operatorname{EM}_q
=
\begin{cases}
\displaystyle
\max_{a\in\mathcal A_q}
\mathbf 1[\nu(\hat y_q)=\nu(a)],
&\mathcal A_q\neq\varnothing,\\[6pt]
0,&\mathcal A_q=\varnothing.
\end{cases}
$$

### 3.2 Token-$F_1$

For $a\in\mathcal A_q$, define the multiset overlap

$$
o_q(a)
=
\left|T_{\nu}(\hat y_q)\cap_{\mathrm{multi}}T_{\nu}(a)\right|.
$$

$$
P_q^{\mathrm{tok}}(a)
=
\begin{cases}
\dfrac{o_q(a)}{|T_{\nu}(\hat y_q)|},&|T_{\nu}(\hat y_q)|>0,\\[6pt]
0,&|T_{\nu}(\hat y_q)|=0,
\end{cases}
\qquad
R_q^{\mathrm{tok}}(a)
=
\begin{cases}
\dfrac{o_q(a)}{|T_{\nu}(a)|},&|T_{\nu}(a)|>0,\\[6pt]
0,&|T_{\nu}(a)|=0,
\end{cases}
$$

$$
F_{1,q}(a)
=
\begin{cases}
\dfrac{
2P_q^{\mathrm{tok}}(a)R_q^{\mathrm{tok}}(a)
}{
P_q^{\mathrm{tok}}(a)+R_q^{\mathrm{tok}}(a)
},
&P_q^{\mathrm{tok}}(a)+R_q^{\mathrm{tok}}(a)>0,\\[6pt]
0,&\text{otherwise},
\end{cases}
$$

$$
\operatorname{TokenF1}_q
=
\begin{cases}
\displaystyle\max_{a\in\mathcal A_q}F_{1,q}(a),
&T_{\nu}(\hat y_q)\neq\varnothing\land\mathcal A_q\neq\varnothing,\\[6pt]
0,&\text{otherwise}.
\end{cases}
$$

### 3.3 ROUGE-L

For the ROUGE scorer tokenization with stemming,

$$
Y_q^{\mathrm R}=\tau_{\mathrm R}(\hat y_q),
\qquad
A_q^{\mathrm R}(a)=\tau_{\mathrm R}(a),
$$

$$
\ell_q(a)=\operatorname{LCS}(Y_q^{\mathrm R},A_q^{\mathrm R}(a)),
$$

$$
P_q^{\mathrm{LCS}}(a)
=
\begin{cases}
\dfrac{\ell_q(a)}{|Y_q^{\mathrm R}|},&|Y_q^{\mathrm R}|>0,\\[6pt]
0,&|Y_q^{\mathrm R}|=0,
\end{cases}
\qquad
R_q^{\mathrm{LCS}}(a)
=
\begin{cases}
\dfrac{\ell_q(a)}{|A_q^{\mathrm R}(a)|},&|A_q^{\mathrm R}(a)|>0,\\[6pt]
0,&|A_q^{\mathrm R}(a)|=0,
\end{cases}
$$

$$
F_{q}^{\mathrm{ROUGE-L}}(a)
=
\begin{cases}
\dfrac{
2P_q^{\mathrm{LCS}}(a)R_q^{\mathrm{LCS}}(a)
}{
P_q^{\mathrm{LCS}}(a)+R_q^{\mathrm{LCS}}(a)
},
&P_q^{\mathrm{LCS}}(a)+R_q^{\mathrm{LCS}}(a)>0,\\[6pt]
0,&\text{otherwise},
\end{cases}
$$

$$
\operatorname{ROUGE-L}_q
=
\begin{cases}
\displaystyle\max_{a\in\mathcal A_q}F_q^{\mathrm{ROUGE-L}}(a),
&\mathcal A_q\neq\varnothing,\\[6pt]
0,&\mathcal A_q=\varnothing.
\end{cases}
$$

### 3.4 Paired-Comparison Sentence BLEU

For the whitespace-tokenized sequences

$$
Y_q^{\mathrm B}=\operatorname{split}(\hat y_q),
\qquad
A_q^{\mathrm B}(a)=\operatorname{split}(a),
$$

let $G_n(Y_q^{\mathrm B})$ be the multiset of prediction $n$-grams for
$n\in\{1,2,3,4\}$. With clipping against all references,

$$
m_{q,n}
=
\sum_{g\in\operatorname{supp}(G_n(Y_q^{\mathrm B}))}
\min\left(
\operatorname{count}_{Y_q^{\mathrm B}}(g),
\max_{a\in\mathcal A_q}\operatorname{count}_{A_q^{\mathrm B}(a)}(g)
\right),
$$

$$
z_{q,n}
=
\max\left(
1,
\sum_{g\in\operatorname{supp}(G_n(Y_q^{\mathrm B}))}
\operatorname{count}_{Y_q^{\mathrm B}}(g)
\right).
$$

The paired script uses NLTK smoothing method 1 with $\varepsilon=0.1$:

$$
\widetilde p_{q,n}
=
\begin{cases}
\dfrac{m_{q,n}}{z_{q,n}},&m_{q,n}>0,\\[6pt]
\dfrac{\varepsilon}{z_{q,n}},&m_{q,n}=0.
\end{cases}
$$

If $c_q=|Y_q^{\mathrm B}|$ and $r_q^{\mathrm{BLEU}}$ is the closest reference
length,

$$
\operatorname{BP}_q
=
\begin{cases}
1,&c_q>r_q^{\mathrm{BLEU}},\\[4pt]
\exp\left(1-\dfrac{r_q^{\mathrm{BLEU}}}{c_q}\right),
&0<c_q\le r_q^{\mathrm{BLEU}},\\[6pt]
0,&c_q=0,
\end{cases}
$$

$$
\operatorname{BLEU}_q
=
\begin{cases}
\displaystyle
\operatorname{BP}_q
\exp\left(
\dfrac{1}{4}\sum_{n=1}^{4}\log\widetilde p_{q,n}
\right),
&c_q>0\land\mathcal A_q\neq\varnothing,\\[8pt]
0,&\text{otherwise}.
\end{cases}
$$

## 4. Semantic and Grounding Metrics

### 4.1 BERTScore-$F_1$

For contextual token embeddings
$H_q=(h_{q,1},\ldots,h_{q,M})$ and
$E_q(a)=(e_{q,1},\ldots,e_{q,L_a})$,

$$
s_{ij}=\frac{h_{q,i}^{\top}e_{q,j}}
{\|h_{q,i}\|_2\|e_{q,j}\|_2},
$$

$$
P_q^{\mathrm{BERT}}(a)
=
\frac{1}{M}\sum_{i=1}^{M}\max_j s_{ij},
\qquad
R_q^{\mathrm{BERT}}(a)
=
\frac{1}{L_a}\sum_{j=1}^{L_a}\max_i s_{ij},
$$

$$
F_q^{\mathrm{BERT}}(a)
=
\frac{
2P_q^{\mathrm{BERT}}(a)R_q^{\mathrm{BERT}}(a)
}{
P_q^{\mathrm{BERT}}(a)+R_q^{\mathrm{BERT}}(a)
},
$$

For the model-specific baseline $b_M$,

$$
\operatorname{Rescale}_{M}(x)
=
\frac{x-b_M}{1-b_M}.
$$

$$
\operatorname{BERTScore}_q
=
\begin{cases}
\displaystyle
\max_{a\in\mathcal A_q}
\operatorname{Rescale}_{M}\left(F_q^{\mathrm{BERT}}(a)\right),
&\hat y_q\neq\varnothing\land\mathcal A_q\neq\varnothing,\\[8pt]
0,&\text{otherwise}.
\end{cases}
$$

### 4.2 Lexical Content-Token Grounding

Let the grounding tokenizer be

$$
\tau_{\mathrm G}(x)
=
\operatorname{Regex}_{\mathtt{[a-z0-9']+}}
\left(\operatorname{lower}(x)\right),
$$

and let $\tau_{\mathrm G}^{\mathrm{content}}$ remove the fixed grounding
stopword set. Then

$$
U_q=\operatorname{set}
\left(\tau_{\mathrm G}^{\mathrm{content}}(\hat y_q)\right),
\qquad
V_q=\bigcup_{c\in\mathcal C_q}\operatorname{set}
\left(\tau_{\mathrm G}(c)\right).
$$

$$
G_q^{\mathrm{lex}}
=
\begin{cases}
1,&U_q=\varnothing,\\[4pt]
0,&U_q\neq\varnothing\land V_q=\varnothing,\\[4pt]
\dfrac{|U_q\cap V_q|}{|U_q|},&\text{otherwise}.
\end{cases}
$$

### 4.3 $n$-gram Grounding

Let $U_q^{(n)}$ be the set of unique contiguous prediction $n$-grams, and
let $V_q^{(n)}$ be the union of within-passage $n$-grams:

$$
U_q^{(n)}=\operatorname{set}
\left(G_n(\tau_{\mathrm G}(\hat y_q))\right),
\qquad
V_q^{(n)}=\bigcup_{c\in\mathcal C_q}\operatorname{set}
\left(G_n(\tau_{\mathrm G}(c))\right).
$$

$$
G_q^{(n)}
=
\begin{cases}
1,&U_q^{(n)}=\varnothing,\\[4pt]
0,&U_q^{(n)}\neq\varnothing\land V_q^{(n)}=\varnothing,\\[4pt]
\dfrac{|U_q^{(n)}\cap V_q^{(n)}|}{|U_q^{(n)}|},&\text{otherwise}.
\end{cases}
$$

The canonical audit uses

$$
n=3.
$$

### 4.4 NLI Entailment Grounding

For

$$
x_q^{\mathrm{premise}}=c_{q,1}\oplus\cdots\oplus c_{q,K},
\qquad
x_q^{\mathrm{hypothesis}}=\hat y_q,
$$

let the NLI classifier logits be

$$
z_q=f_{\theta}
\left(x_q^{\mathrm{premise}},x_q^{\mathrm{hypothesis}}\right).
$$

Then

$$
G_q^{\mathrm{NLI}}
=
\begin{cases}
0,&\hat y_q=\varnothing\ \lor\ x_q^{\mathrm{premise}}=\varnothing,\\[4pt]
\dfrac{\exp(z_{q,\mathrm{ent}})}
{\sum_{\ell\in\{\mathrm{ent},\mathrm{neu},\mathrm{con}\}}
\exp(z_{q,\ell})},
&\text{otherwise},
\end{cases}
$$

where $\mathrm{ent}$, $\mathrm{neu}$, and $\mathrm{con}$ denote entailment,
neutral, and contradiction.

## 5. RAG Triad

With qrels available,

$$
C_q^{\mathrm{rel}}
=
\mathbf 1
\left[
\{d_{q,1},\ldots,d_{q,K}\}\cap R_q\neq\varnothing
\right].
$$

Without qrels,

$$
C_q^{\mathrm{rel}}
=
G^{\mathrm{lex}}(\text{query}_q,\mathcal C_q).
$$

The implemented diagnostic vector is

$$
\mathbf t_q
=
\left(
C_q^{\mathrm{rel}},
G_q^{\mathrm{lex}},
\operatorname{TokenF1}_q
\right),
$$

and its convenience summary is

$$
\operatorname{Triad}_q
=
\frac{1}{3}
\left(
C_q^{\mathrm{rel}}
+G_q^{\mathrm{lex}}
+\operatorname{TokenF1}_q
\right).
$$

## 6. Paired Comparison

For two systems $A$ and $B$ evaluated on the same ordered query set,

$$
m_i^{A}=m(\hat y_i^{A}),
\qquad
m_i^{B}=m(\hat y_i^{B}),
\qquad
d_i=m_i^{B}-m_i^{A}.
$$

The reported point estimates are

$$
\widehat\mu_A=\frac{1}{N}\sum_{i=1}^{N}m_i^{A},
\qquad
\widehat\mu_B=\frac{1}{N}\sum_{i=1}^{N}m_i^{B},
$$

$$
\widehat\Delta
=
\widehat\mu_B-\widehat\mu_A
=
\frac{1}{N}\sum_{i=1}^{N}d_i.
$$

## 7. Paired Bootstrap Confidence Interval

For bootstrap replicate $b\in\{1,\ldots,B\}$, draw

$$
I_{b,1},\ldots,I_{b,N}
\overset{\mathrm{iid}}{\sim}
\operatorname{Uniform}\{1,\ldots,N\},
$$

and compute

$$
\Delta_b^{*}
=
\frac{1}{N}\sum_{j=1}^{N}d_{I_{b,j}}.
$$

For confidence level $1-\alpha$,

$$
\operatorname{CI}_{1-\alpha}
=
\left[
Q_{\alpha/2}\left(\{\Delta_b^{*}\}_{b=1}^{B}\right),
Q_{1-\alpha/2}\left(\{\Delta_b^{*}\}_{b=1}^{B}\right)
\right].
$$

The repository defaults are

$$
B=10{,}000,
\qquad
\alpha=0.05,
\qquad
\text{seed}=42.
$$

## 8. Reported Two-Sided Bootstrap Tail Probability

The implementation counts both inclusive tails of the empirical paired
bootstrap distribution:

$$
n_{\le 0}
=
\sum_{b=1}^{B}\mathbf 1[\Delta_b^{*}\le 0],
\qquad
n_{\ge 0}
=
\sum_{b=1}^{B}\mathbf 1[\Delta_b^{*}\ge 0].
$$

The reported value is

$$
\boxed{
p_{\mathrm{boot-tail}}
=
\min\left(
1,
2\min\left(
\frac{n_{\le 0}}{B},
\frac{n_{\ge 0}}{B}
\right)
\right)
}.
$$

Thus

$$
p_{\mathrm{reported}}
\equiv
p_{\mathrm{boot-tail}},
$$

not a null-centered paired-permutation $p$-value. The percentile confidence
interval is the primary inferential quantity in the current experiment
contract.

## 9. Length-Control Diagnostics

This design diagnostic is not yet emitted by the current runners. Output
length is a covariate, not a replacement quality metric:

$$
\lambda_i^{S}=|T_{\nu}(\hat y_i^{S})|,
\qquad
\Delta\lambda_i=\lambda_i^{B}-\lambda_i^{A},
$$

$$
\overline{\Delta\lambda}
=
\frac{1}{N}\sum_{i=1}^{N}\Delta\lambda_i,
$$

$$
\rho_{m,\lambda}
=
\operatorname{corr}
\left(
\{d_i\}_{i=1}^{N},
\{\Delta\lambda_i\}_{i=1}^{N}
\right).
$$

The same paired-bootstrap construction applies to
$\overline{\Delta\lambda}$.
